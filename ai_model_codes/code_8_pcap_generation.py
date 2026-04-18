import sys
import time
import random
import itertools
import numpy as np
import pandas as pd
from pathlib import Path

try:
    from scapy.all import Ether, IP, TCP, Raw, wrpcap
except ImportError:
    print("❌ Scapy not installed. Run: pip install scapy")
    sys.exit(1)

# NETWORK TOPOLOGY — real IPs from Wireshark capture
BROKER_IP   = "192.168.0.133"
PI_IP       = "192.168.0.109"
MQTT_PORT   = 1883

IOT_DEVICES = [
    "192.168.0.153",   # ESP32 Node 1: hazards
    "192.168.0.160",   # ESP32 Node 2: traffic
    "192.168.0.176",   # ESP32 Node 3: environment + bridge
]

PI_MAC     = "02:00:00:00:00:01"   # ← replace before Pi replay
BROKER_MAC = "ff:ff:ff:ff:ff:ff"   # ← replace before Pi replay (broadcast works too)

EPH_PORT_START = 49152
EPH_PORT_END   = 65535

# PATHS — Windows laptop pointing at your existing CSV output

BASE_DIR = Path(r"C:\Users\Lenovo\Downloads\DoS-DDoS-MQTT-IoT_Dataset")

CSV_DIR        = BASE_DIR / "FINALCTGAN_Synthetic_Traffic_v2" / "per_attack"
OUTPUT_DIR     = BASE_DIR / "NeuroStrike_HW_PCAPs"
PCAP_PURE_DIR  = OUTPUT_DIR / "pure"
PCAP_BLEND_DIR = OUTPUT_DIR / "blended"

PCAP_PURE_DIR.mkdir(parents=True, exist_ok=True)
PCAP_BLEND_DIR.mkdir(parents=True, exist_ok=True)

# ATTACK CONFIGURATION
ATTACK_TYPES = [
    "Basic_Connect_Flooding",
    "Connect_Flooding_with_WILL_payload",
    "Delayed_Connect_Flooding",
    "Invalid_Subscription_Flooding",
    "SYN_TCP_Flooding",
]

SHORT_NAMES = {
    "Basic_Connect_Flooding":             "Basic",
    "Connect_Flooding_with_WILL_payload": "WILL",
    "Delayed_Connect_Flooding":           "Delayed",
    "Invalid_Subscription_Flooding":      "Invalid",
    "SYN_TCP_Flooding":                   "SYN",
}

BLEND_RATIOS = [0.30, 0.50, 0.70]

PACKETS_PER_PURE  = 300_000
PACKETS_PER_BLEND = 50_000

# FLOW TRACKER
class FlowTracker:

    def __init__(self):
        self.seq = {}

    def get_seq(self, key: tuple) -> int:
        if key not in self.seq:
            self.seq[key] = random.randint(1_000_000, 3_000_000_000)
        return self.seq[key]

    def advance(self, key: tuple, payload_len: int):
        self.seq[key] = (self.seq[key] + max(payload_len, 1)) % (2 ** 32)

    def get_ack(self, key: tuple) -> int:
        reverse = (key[1], key[0], key[3], key[2])
        return self.seq.get(reverse, 0)

# FLAG BUILDER
def build_flags(row) -> str:
    flags = ""
    if int(row.get("flag_syn", 0)): flags += "S"
    if int(row.get("flag_ack", 0)): flags += "A"
    if int(row.get("flag_fin", 0)): flags += "F"
    if int(row.get("flag_rst", 0)): flags += "R"
    if int(row.get("flag_psh", 0)): flags += "P"
    if int(row.get("flag_urg", 0)): flags += "U"
    return flags if flags else "A"

# ROW → PACKET
def row_to_packet(row,
                  timestamp: float,
                  flow_tracker: FlowTracker,
                  device_ports: dict,
                  packet_idx: int):

    port_dir    = int(row.get("port_direction", 1))
    payload_len = max(0, int(row.get("payload_len", 0)))
    win_size    = max(0, min(int(row.get("tcp_window_size", 65535)), 65535))

    device_ip = IOT_DEVICES[packet_idx % len(IOT_DEVICES)]
    if device_ip not in device_ports:
        device_ports[device_ip] = random.randint(EPH_PORT_START, EPH_PORT_END)
    device_port = device_ports[device_ip]

    if port_dir == 1:
        src_ip, dst_ip     = device_ip, BROKER_IP
        src_port, dst_port = device_port, MQTT_PORT
    else:
        src_ip, dst_ip     = BROKER_IP, device_ip
        src_port, dst_port = MQTT_PORT, device_port

    flow_key = (src_ip, dst_ip, src_port, dst_port)
    seq      = flow_tracker.get_seq(flow_key)
    ack      = flow_tracker.get_ack(flow_key)

    payload = (bytes(random.getrandbits(8) for _ in range(payload_len))
               if payload_len > 0 else b"")

    pkt = (
        Ether(src=PI_MAC, dst=BROKER_MAC) /
        IP(src=src_ip, dst=dst_ip) /
        TCP(
            sport=src_port,
            dport=dst_port,
            flags=build_flags(row),
            seq=seq,
            ack=ack,
            window=win_size,
        ) /
        Raw(load=payload)
    )
    pkt.time = timestamp
    flow_tracker.advance(flow_key, payload_len)
    return pkt

# GENERATE PURE
def generate_pure(attack_name: str,
                  df: pd.DataFrame,
                  pcap_path: Path,
                  n_packets: int) -> tuple:

    if len(df) > n_packets:
        df = df.head(n_packets).copy()
    df = df.reset_index(drop=True)
    print(f"  Rows: {len(df):,}")

    flow_tracker = FlowTracker()
    device_ports = {}
    packets      = []
    timestamp    = 0.0
    t0           = time.time()

    for idx, row in df.iterrows():
        timestamp += max(0.0, float(row.get("delta_time", 0)))
        packets.append(row_to_packet(
            row, timestamp, flow_tracker, device_ports, idx))
        if (idx + 1) % 10_000 == 0:
            _progress(idx + 1, len(df), t0)

    _write(packets, pcap_path, timestamp, t0)
    return len(packets), timestamp

# GENERATE BLEND
def generate_blend(name_a: str,
                   name_b: str,
                   df_a: pd.DataFrame,
                   df_b: pd.DataFrame,
                   alpha: float,
                   pcap_path: Path,
                   n_packets: int) -> tuple:

    rng     = np.random.default_rng(42)
    choices = rng.random(n_packets) < alpha

    idx_a   = rng.integers(0, len(df_a), size=int(choices.sum()))
    idx_b   = rng.integers(0, len(df_b), size=int((~choices).sum()))
    rows_a  = df_a.iloc[idx_a].reset_index(drop=True)
    rows_b  = df_b.iloc[idx_b].reset_index(drop=True)

    blended = []
    iter_a  = iter(rows_a.itertuples(index=False))
    iter_b  = iter(rows_b.itertuples(index=False))
    for use_a in choices:
        try:
            row = next(iter_a) if use_a else next(iter_b)
            blended.append(row._asdict())
        except StopIteration:
            break

    print(f"  Blend : {alpha:.0%} {SHORT_NAMES[name_a]} "
          f"+ {1-alpha:.0%} {SHORT_NAMES[name_b]}")
    print(f"  Rows  : {len(blended):,}  "
          f"(A={int(choices.sum()):,}  B={int((~choices).sum()):,})")

    flow_tracker = FlowTracker()
    device_ports = {}
    packets      = []
    timestamp    = 0.0
    t0           = time.time()

    for idx, row in enumerate(blended):
        timestamp += max(0.0, float(row.get("delta_time", 0)))
        packets.append(row_to_packet(
            row, timestamp, flow_tracker, device_ports, idx))
        if (idx + 1) % 10_000 == 0:
            _progress(idx + 1, len(blended), t0)

    _write(packets, pcap_path, timestamp, t0)
    return len(packets), timestamp

# HELPERS
def _progress(done: int, total: int, t0: float):
    elapsed = time.time() - t0
    pps     = done / max(elapsed, 0.001)
    eta     = (total - done) / max(pps, 1)
    print(f"  [{done:>7,}/{total:,}]  "
          f"{elapsed:>5.0f}s  ETA {eta:>4.0f}s  "
          f"{pps:>6.0f} pkt/s")


def _write(packets: list, pcap_path: Path, duration: float, t0: float):
    print(f"  Writing {len(packets):,} packets...")
    wrpcap(str(pcap_path), packets)
    size_mb = pcap_path.stat().st_size / 1e6
    elapsed = time.time() - t0
    print(f"  ✓ {pcap_path.name}  "
          f"({size_mb:.1f} MB)  "
          f"{elapsed:.1f}s  "
          f"replay={duration:.2f}s")

# MAIN
def main():
    print("=" * 65)
    print("NeuroStrike - Code 8: HW-Ready PCAP Generator")
    print("=" * 65)
    print(f"\nNetwork topology (from Wireshark capture):")
    print(f"  Broker    : {BROKER_IP}:{MQTT_PORT}")
    print(f"  Pi sender : {PI_IP}")
    for i, ip in enumerate(IOT_DEVICES):
        print(f"  IoT Node {i+1}: {ip}  (spoofed source)")
    print(f"\nMACs:")
    print(f"  Pi MAC     : {PI_MAC}")
    print(f"  Broker MAC : {BROKER_MAC}")
    if BROKER_MAC == "ff:ff:ff:ff:ff:ff":
        print(f"    Broadcast MAC in use — update before Pi replay")
        print(f"      Run on Pi: arp -a {BROKER_IP}")
    print(f"\nCSV dir  : {CSV_DIR}")
    print(f"Output   : {OUTPUT_DIR}")

    if not CSV_DIR.exists():
        print(f"\n CSV directory not found: {CSV_DIR}")
        print(f"   Check BASE_DIR path at top of script.")
        sys.exit(1)

    # Load CSVs
    print(f"\nLoading CSVs...")
    dataframes = {}
    for attack in ATTACK_TYPES:
        p = CSV_DIR / f"{attack}_synthetic.csv"
        if p.exists():
            df = pd.read_csv(p)
            dataframes[attack] = df
            print(f"  ✓ {attack}  "
                  f"({len(df):,} rows  {p.stat().st_size/1e6:.1f} MB)")
        else:
            print(f"    Missing: {attack}_synthetic.csv")

    if not dataframes:
        print(f"\n No CSVs found.")
        sys.exit(1)

    n_pairs  = len(list(itertools.combinations(dataframes.keys(), 2)))
    n_blends = n_pairs * len(BLEND_RATIOS)
    n_total  = len(dataframes) + n_blends

    print(f"\nPlan:")
    print(f"  Pure    : {len(dataframes)} PCAPs  "
          f"({PACKETS_PER_PURE:,} packets each)")
    print(f"  Blended : {n_blends} PCAPs  "
          f"({n_pairs} pairs × {len(BLEND_RATIOS)} ratios, "
          f"{PACKETS_PER_BLEND:,} packets each)")
    print(f"  Total   : {n_total} PCAPs")

    total_start   = time.time()
    results_pure  = {}
    results_blend = {}

    # Pure attacks
    print(f"\n{'='*65}")
    print(f"CATEGORY 1: Pure Attacks")
    print(f"{'='*65}")

    for attack, df in dataframes.items():
        pcap_path = PCAP_PURE_DIR / f"{attack}.pcap"
        print(f"\n  {'─'*60}")
        print(f"  {attack}")
        print(f"  {'─'*60}")
        n, dur = generate_pure(attack, df.copy(), pcap_path, PACKETS_PER_PURE)
        results_pure[attack] = (n, dur)

    # Blended attacks
    pairs     = list(itertools.combinations(dataframes.keys(), 2))
    blend_num = 0

    print(f"\n{'='*65}")
    print(f"CATEGORY 2: Blended Attacks  ({n_blends} PCAPs)")
    print(f"{'='*65}")

    for attack_a, attack_b in pairs:
        short_a = SHORT_NAMES[attack_a]
        short_b = SHORT_NAMES[attack_b]

        for alpha in BLEND_RATIOS:
            blend_num += 1
            ratio_str = f"{int(alpha*100)}_{int((1-alpha)*100)}"
            fname     = f"{short_a}_x_{short_b}_{ratio_str}.pcap"
            pcap_path = PCAP_BLEND_DIR / fname

            print(f"\n  {'─'*60}")
            print(f"  [{blend_num}/{n_blends}] "
                  f"{short_a} × {short_b}  "
                  f"{int(alpha*100)}/{int((1-alpha)*100)}")
            print(f"  {'─'*60}")

            n, dur = generate_blend(
                attack_a, attack_b,
                dataframes[attack_a].copy(),
                dataframes[attack_b].copy(),
                alpha, pcap_path,
                PACKETS_PER_BLEND
            )
            results_blend[fname] = (n, dur)

    total_elapsed = time.time() - total_start

    # Summary 
    print(f"\n{'='*65}")
    print(f" COMPLETE  ({total_elapsed:.0f}s total)")
    print(f"{'='*65}")

    total_size = 0
    print(f"\n  Pure ({len(results_pure)}):")
    for attack, (n, dur) in results_pure.items():
        p    = PCAP_PURE_DIR / f"{attack}.pcap"
        size = p.stat().st_size / 1e6
        total_size += size
        print(f"    {attack:<45} {n:>7,} pkts  "
              f"{dur:>7.1f}s replay  {size:.1f} MB")

    print(f"\n  Blended ({len(results_blend)}):")
    for fname, (n, dur) in results_blend.items():
        p    = PCAP_BLEND_DIR / fname
        size = p.stat().st_size / 1e6
        total_size += size
        print(f"    {fname:<50} {n:>7,} pkts  {size:.1f} MB")

    print(f"\n  Total output size : {total_size:.1f} MB")
    print(f"  Pure dir          : {PCAP_PURE_DIR}")
    print(f"  Blended dir       : {PCAP_BLEND_DIR}")

    print(f"\n{'='*65}")
    print(f"BEFORE REPLAYING ON PI")
    print(f"{'='*65}")
    print(f"\n  1. Get real MACs (run on Pi):")
    print(f"       ip link show wlan0          ← Pi MAC")
    print(f"       arp -a {BROKER_IP}   ← Broker MAC")
    print(f"\n  2. Update MACs in PCAPs (run on Pi after scp):")
    print(f"       for f in pure/*.pcap blended/*.pcap; do")
    print(f"         tcprewrite \\")
    print(f"           --enet-smac=<pi_mac> \\")
    print(f"           --enet-dmac=<broker_mac> \\")
    print(f"           --infile=$f --outfile=$f")
    print(f"       done")
    print(f"\n  3. Transfer to Pi:")
    print(f"       scp -r {OUTPUT_DIR} pi@{PI_IP}:/home/pi/neuro_strike/")
    print(f"\n  4. Replay on Pi:")
    print(f"       sudo tcpreplay --intf=wlan0 --timing=nano \\")
    print(f"         /home/pi/neuro_strike/NeuroStrike_HW_PCAPs/pure/<attack>.pcap")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
