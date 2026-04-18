import os
import glob
import re
import pandas as pd

# CONFIG
ATTACK_INPUT_FOLDERS = {
    "Basic Connect Flooding":              r"C:\Users\Lenovo\Downloads\DoS-DDoS-MQTT-IoT_Dataset\Basic Connect Flooding\CSV_converted",
    "Connect Flooding with WILL payload":  r"C:\Users\Lenovo\Downloads\DoS-DDoS-MQTT-IoT_Dataset\Connect Flooding with WILL payload\CSV_converted",
    "Delayed Connect Flooding":            r"C:\Users\Lenovo\Downloads\DoS-DDoS-MQTT-IoT_Dataset\Delayed Connect Flooding\CSV_converted",
    "Invalid Subscription Flooding":       r"C:\Users\Lenovo\Downloads\DoS-DDoS-MQTT-IoT_Dataset\Invalid Subscription Flooding\CSV_converted",
    "SYN TCP Flooding":                    r"C:\Users\Lenovo\Downloads\DoS-DDoS-MQTT-IoT_Dataset\SYN TCP Flooding\CSV_converted",
}

OUTPUT_ROOT = r"C:\Users\Lenovo\Downloads\DoS-DDoS-MQTT-IoT_Dataset\Cleaned_Per_PCAP"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# MQTT ports (standard + TLS)
MQTT_PORTS = {1883, 8883}


# PORT EXTRACTION
def extract_ports(row):
    src_port, dst_port = None, None

    src_candidates = ['Source Port', 'Src Port', 'src_port', 'srcport']
    dst_candidates = ['Destination Port', 'Dst Port', 'dst_port', 'dstport']

    for col in src_candidates:
        if col in row.index and pd.notna(row[col]):
            src_port = row[col]
            break

    for col in dst_candidates:
        if col in row.index and pd.notna(row[col]):
            dst_port = row[col]
            break

    if (src_port is None or dst_port is None) and 'Info' in row.index:
        info = row['Info']
        if isinstance(info, str):
            match = re.search(r'(\d+)\s*(?:>|→|->)\s*(\d+)', info)
            if match:
                if src_port is None:
                    src_port = int(match.group(1))
                if dst_port is None:
                    dst_port = int(match.group(2))


    try:
        src_port = int(src_port) if src_port is not None and pd.notna(src_port) else None
        dst_port = int(dst_port) if dst_port is not None and pd.notna(dst_port) else None
    except (ValueError, TypeError):
        return None, None

    return src_port, dst_port


def get_port_features(row):
    src_port, dst_port = extract_ports(row)

    dst_is_mqtt = dst_port in MQTT_PORTS if dst_port is not None else False
    src_is_mqtt = src_port in MQTT_PORTS if src_port is not None else False

    is_mqtt = dst_is_mqtt or src_is_mqtt

    if dst_is_mqtt:
        direction = 1   # Packet going TO broker
    elif src_is_mqtt:
        direction = 0   # Packet coming FROM broker
    else:
        direction = -1  # Unknown (will be filtered out anyway)

    return is_mqtt, direction


# TCP FLAG EXTRACTION
FLAG_PATTERNS = {
    'flag_syn': re.compile(r'\bSYN\b', re.I),
    'flag_ack': re.compile(r'\bACK\b', re.I),
    'flag_fin': re.compile(r'\bFIN\b', re.I),
    'flag_rst': re.compile(r'\bRST\b', re.I),
    'flag_psh': re.compile(r'\bPSH\b', re.I),
    'flag_urg': re.compile(r'\bURG\b', re.I),
}


def extract_tcp_flags(info: str) -> dict:
    flags = {flag: 0 for flag in FLAG_PATTERNS}

    if not isinstance(info, str):
        return flags

    for flag_name, pattern in FLAG_PATTERNS.items():
        if pattern.search(info):
            flags[flag_name] = 1

    return flags


# PAYLOAD LENGTH EXTRACTION
def extract_payload_len(row) -> float:
    payload_cols = ['Payload Length', 'payload_len', 'TCP Payload', 'tcp_payload_len', 'Payload']
    for col in payload_cols:
        if col in row.index and pd.notna(row[col]):
            try:
                return float(row[col])
            except (ValueError, TypeError):
                pass

    if 'Info' in row.index and isinstance(row['Info'], str):
        match = re.search(r'\bLen=(\d+)', row['Info'], re.I)
        if match:
            return float(match.group(1))

    return 0.0


# TCP WINDOW SIZE EXTRACTION
def extract_tcp_window_size(row) -> float:
    win_cols = ['Window Size', 'window_size', 'TCP Window Size', 'tcp_window', 'Win']
    for col in win_cols:
        if col in row.index and pd.notna(row[col]):
            try:
                return float(row[col])
            except (ValueError, TypeError):
                pass

    if 'Info' in row.index and isinstance(row['Info'], str):
        match = re.search(r'\bWin=(\d+)', row['Info'], re.I)
        if match:
            return float(match.group(1))

    return 0.0


# CLEAN ONE CSV (ONE PCAP)
def clean_one_csv(csv_path: str, out_dir: str, idx: int) -> dict:
    stats = {
        'success':              False,
        'total_packets':        0,
        'tcp_packets':          0,
        'mqtt_port_packets':    0,
        'error':                None
    }

    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]

        # VALIDATE REQUIRED COLUMNS
        required = ["Time", "Protocol", "Length"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            stats['error'] = f"Missing columns: {missing}"
            return stats

        # TYPE SAFETY
        df["Time"]      = pd.to_numeric(df["Time"],   errors="coerce")
        df["packet_len"] = pd.to_numeric(df["Length"], errors="coerce")
        df["Protocol"]  = df["Protocol"].astype(str).str.strip().str.upper()

        df = df.dropna(subset=["Time", "packet_len"])
        df = df[df["packet_len"] > 0].copy()

        stats['total_packets'] = len(df)

        # TCP ONLY
        df = df[df["Protocol"] == "TCP"].copy()
        if df.empty:
            stats['error'] = "No TCP packets found"
            return stats

        stats['tcp_packets'] = len(df)

        # MQTT PORT FILTER + PORT DIRECTION
        port_results    = df.apply(get_port_features, axis=1)
        df['is_mqtt']   = port_results.apply(lambda x: x[0])
        df['port_direction'] = port_results.apply(lambda x: x[1])

        df = df[df['is_mqtt'] == True].copy()
        if df.empty:
            stats['error'] = "No TCP traffic on MQTT ports (1883/8883)"
            return stats

        stats['mqtt_port_packets'] = len(df)

        # SORT + DELTA TIME (PER FILE)
        df = df.sort_values("Time").reset_index(drop=True)
        df["delta_time"] = df["Time"].diff().fillna(0.0)
        df.loc[df["delta_time"] < 0, "delta_time"] = 0.0  # Safety clamp

        # TCP FLAGS (from Info column)
        if 'Info' in df.columns:
            flags_df = df['Info'].apply(extract_tcp_flags).apply(pd.Series)
        else:
            flags_df = pd.DataFrame(
                {flag: 0 for flag in FLAG_PATTERNS},
                index=df.index
            )
        df = pd.concat([df, flags_df], axis=1)

        # PAYLOAD LENGTH
        df["payload_len"] = df.apply(extract_payload_len, axis=1)

        # TCP WINDOW SIZE
        df["tcp_window_size"] = df.apply(extract_tcp_window_size, axis=1)

        # FINAL 11-FEATURE OUTPUT
        out = df[[
            "delta_time",       # 1
            "packet_len",       # 2
            "flag_syn",         # 3
            "flag_ack",         # 4
            "flag_fin",         # 5
            "flag_rst",         # 6
            "flag_psh",         # 7
            "flag_urg",         # 8
            "port_direction",   # 9
            "payload_len",      # 10
            "tcp_window_size",  # 11
        ]].copy()

        # SAVE PER-PCAP OUTPUT
        out_name = f"cleaned_pcap_{idx:03d}.csv"
        out_path = os.path.join(out_dir, out_name)
        out.to_csv(out_path, index=False)

        stats['success'] = True
        return stats

    except Exception as e:
        stats['error'] = str(e)
        return stats


# MAIN
def main():
    print("=" * 70)
    print("NEUROSTRIKE - CODE 1: TCP/MQTT PREPROCESSING")
    print("=" * 70)
    print("\nFeatures (11 total):")
    print("  1.  delta_time       - Inter-packet timing")
    print("  2.  packet_len       - Total packet size in bytes")
    print("  3.  flag_syn         - SYN flag")
    print("  4.  flag_ack         - ACK flag")
    print("  5.  flag_fin         - FIN flag")
    print("  6.  flag_rst         - RST flag")
    print("  7.  flag_psh         - PSH flag")
    print("  8.  flag_urg         - URG flag")
    print("  9.  port_direction   - 1=toward broker, 0=from broker")
    print("  10. payload_len      - Actual data payload in bytes")
    print("  11. tcp_window_size  - TCP window size")
    print("\nFilter: TCP only | Ports: 1883 or 8883 (source OR destination)\n")

    # Global statistics
    global_stats = {
        'total_files':           0,
        'successful':            0,
        'failed':                0,
        'total_packets':         0,
        'total_tcp_packets':     0,
        'total_mqtt_packets':    0,
        'attack_breakdown':      {}
    }

    for attack_name, folder in ATTACK_INPUT_FOLDERS.items():
        out_dir = os.path.join(OUTPUT_ROOT, attack_name.replace(" ", "_"))
        os.makedirs(out_dir, exist_ok=True)

        csv_files = sorted(glob.glob(os.path.join(folder, "*.csv")))

        if not csv_files:
            print(f"⚠️  [{attack_name}] No CSV files found\n")
            continue

        print(f"[{attack_name}] Processing {len(csv_files)} files...")

        attack_stats = {'files': 0, 'successful': 0, 'mqtt_packets': 0}

        for i, csv_path in enumerate(csv_files):
            global_stats['total_files'] += 1
            attack_stats['files'] += 1

            stats = clean_one_csv(csv_path, out_dir, i)

            if stats['success']:
                global_stats['successful']         += 1
                attack_stats['successful']          += 1
                global_stats['total_packets']       += stats['total_packets']
                global_stats['total_tcp_packets']   += stats['tcp_packets']
                global_stats['total_mqtt_packets']  += stats['mqtt_port_packets']
                attack_stats['mqtt_packets']         += stats['mqtt_port_packets']
                print(f"  ✓ {os.path.basename(csv_path)}: {stats['mqtt_port_packets']:,} packets")
            else:
                global_stats['failed'] += 1
                print(f"  ✗ {os.path.basename(csv_path)}: {stats['error']}")

        global_stats['attack_breakdown'][attack_name] = attack_stats
        print(f"  → {attack_stats['successful']}/{attack_stats['files']} successful\n")

    # SUMMARY REPORT
    print("=" * 70)
    print("✅ PROCESSING COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory : {OUTPUT_ROOT}")
    print(f"Total files      : {global_stats['total_files']}")
    print(f"Successful       : {global_stats['successful']}")
    print(f"Failed/Skipped   : {global_stats['failed']}")
    print(f"Total packets    : {global_stats['total_packets']:,}")
    print(f"TCP packets      : {global_stats['total_tcp_packets']:,}")
    print(f"MQTT port packets: {global_stats['total_mqtt_packets']:,}")

    print(f"\nPer-Attack Breakdown:")
    for attack_name, stats in global_stats['attack_breakdown'].items():
        print(f"  {attack_name}:")
        print(f"    Files  : {stats['successful']}/{stats['files']}")
        print(f"    Packets: {stats['mqtt_packets']:,}")

    # SAVE SUMMARY FILE
    summary_path = os.path.join(OUTPUT_ROOT, "processing_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("NEUROSTRIKE - CODE 1: PREPROCESSING SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write("FEATURES (11 TOTAL)\n")
        f.write("-" * 70 + "\n")
        f.write("1.  delta_time       - Inter-packet timing (per-file, no leakage)\n")
        f.write("2.  packet_len       - Total packet size in bytes\n")
        f.write("3.  flag_syn         - SYN flag (binary)\n")
        f.write("4.  flag_ack         - ACK flag (binary)\n")
        f.write("5.  flag_fin         - FIN flag (binary)\n")
        f.write("6.  flag_rst         - RST flag (binary)\n")
        f.write("7.  flag_psh         - PSH flag (binary)\n")
        f.write("8.  flag_urg         - URG flag (binary)\n")
        f.write("9.  port_direction   - 1=toward broker, 0=from broker\n")
        f.write("10. payload_len      - TCP payload length in bytes\n")
        f.write("11. tcp_window_size  - TCP window size\n\n")

        f.write("FILTER CRITERIA\n")
        f.write("-" * 70 + "\n")
        f.write("Protocol : TCP only\n")
        f.write("Ports    : 1883 or 8883 (source OR destination)\n\n")

        f.write("STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total files processed : {global_stats['total_files']}\n")
        f.write(f"Successful            : {global_stats['successful']}\n")
        f.write(f"Failed                : {global_stats['failed']}\n")
        f.write(f"Total packets         : {global_stats['total_packets']:,}\n")
        f.write(f"TCP packets           : {global_stats['total_tcp_packets']:,}\n")
        f.write(f"MQTT port packets     : {global_stats['total_mqtt_packets']:,}\n\n")

        f.write("PER-ATTACK BREAKDOWN\n")
        f.write("-" * 70 + "\n")
        for attack_name, stats in global_stats['attack_breakdown'].items():
            f.write(f"\n{attack_name}:\n")
            f.write(f"  Files  : {stats['successful']}/{stats['files']}\n")
            f.write(f"  Packets: {stats['mqtt_packets']:,}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("LEAKAGE PREVENTION\n")
        f.write("=" * 70 + "\n")
        f.write("No IP addresses in output\n")
        f.write("No TCP sequence numbers\n")
        f.write("No raw port numbers in output\n")
        f.write("No flow identifiers\n")
        f.write("Per-file delta_time (no cross-file timing leakage)\n")
        f.write("Negative delta_time clamped to 0\n")
        f.write("Independent per-file processing\n")


    print(f"\n✓ Summary saved to: processing_summary.txt")


if __name__ == "__main__":
    main()
