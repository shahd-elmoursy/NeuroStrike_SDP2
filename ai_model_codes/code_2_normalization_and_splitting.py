import os
import glob
import json
import gc
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

# CONFIGURATION

BASE_DIR   = Path(r"C:\Users\Lenovo\Downloads\DoS-DDoS-MQTT-IoT_Dataset")
INPUT_DIR  = BASE_DIR / "Cleaned_Per_PCAP"
OUTPUT_DIR = BASE_DIR / "CTGAN_Ready_NeuroStrike_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PER_ATTACK_DIR = OUTPUT_DIR / "per_attack"
PER_ATTACK_DIR.mkdir(exist_ok=True)

ATTACK_TYPES = [
    "Basic_Connect_Flooding",
    "Connect_Flooding_with_WILL_payload",
    "Delayed_Connect_Flooding",
    "Invalid_Subscription_Flooding",
    "SYN_TCP_Flooding",
]

# Attacks where payload_len is meaningfully non-zero — apply log transform
PAYLOAD_LOG_ATTACKS = {
    "Connect_Flooding_with_WILL_payload",
    "Invalid_Subscription_Flooding",
}

CONTINUOUS_FEATURES = ["delta_time", "packet_len", "payload_len", "tcp_window_size"]
BINARY_FEATURES     = ["flag_syn", "flag_ack", "flag_fin", "flag_rst",
                        "flag_psh", "flag_urg", "port_direction"]
ALL_FEATURES        = CONTINUOUS_FEATURES + BINARY_FEATURES
NUM_FEATURES        = len(ALL_FEATURES)

MAX_PACKETS_PER_ATTACK = 500_000
RANDOM_SEED            = 42

# Split ratios
TEST_RATIO  = 0.15          # 15% held out for final evaluation
VAL_RATIO   = 0.176         # 0.176 of remaining 85% ≈ 15% of total
# Result: ~70% train, ~15% val, ~15% test

# HELPERS
def elapsed(start: float) -> str:
    s = time.time() - start
    if s < 60:
        return f"{s:.1f}s"
    return f"{int(s//60)}m {int(s%60)}s"


def verify_no_file_overlap(train_df, val_df, test_df):
    train_files = set(train_df["_file_name"].unique())
    val_files   = set(val_df["_file_name"].unique())
    test_files  = set(test_df["_file_name"].unique())

    tv = train_files & val_files
    tt = train_files & test_files
    vt = val_files   & test_files

    assert len(tv) == 0, f"LEAKAGE: {len(tv)} files in both train and val: {tv}"
    assert len(tt) == 0, f"LEAKAGE: {len(tt)} files in both train and test: {tt}"
    assert len(vt) == 0, f"LEAKAGE: {len(vt)} files in both val and test: {vt}"

    print(f"    No file leakage confirmed across all three splits")
    print(f"    Train files : {len(train_files)}")
    print(f"    Val files   : {len(val_files)}")
    print(f"    Test files  : {len(test_files)}")


# LOG TRANSFORMS
def apply_log_transforms(df: pd.DataFrame, attack_name: str) -> pd.DataFrame:
    df = df.copy()
    df["delta_time"] = np.log1p(df["delta_time"].values.astype(float) * 1000)
    if attack_name in PAYLOAD_LOG_ATTACKS:
        df["payload_len"] = np.log1p(df["payload_len"].values.astype(float))
    return df


# LOAD ONE ATTACK TYPE
def load_attack_type(attack_name: str,
                     max_packets: int,
                     seed: int) -> pd.DataFrame:
    attack_dir = INPUT_DIR / attack_name
    if not attack_dir.exists():
        print(f"    Directory not found: {attack_dir}")
        return pd.DataFrame()

    csv_files = sorted(glob.glob(str(attack_dir / "cleaned_pcap_*.csv")))
    if not csv_files:
        print(f"    No CSV files in: {attack_dir}")
        return pd.DataFrame()

    print(f"    Found {len(csv_files)} PCAP files")

    # Count rows per file
    file_lengths = {}
    t0 = time.time()
    for i, f in enumerate(csv_files):
        try:
            with open(f, 'r') as fh:
                n = sum(1 for _ in fh) - 1
            file_lengths[f] = max(0, n)
        except Exception:
            file_lengths[f] = 0
        if (i + 1) % 10 == 0:
            print(f"    Counted {i+1}/{len(csv_files)} files... ({elapsed(t0)})")

    total_available = sum(file_lengths.values())
    print(f"    Total available packets: {total_available:,}")
    if total_available == 0:
        return pd.DataFrame()

    # Proportional per-file quota
    quota = {f: max(1, int(n / total_available * max_packets))
             for f, n in file_lengths.items() if n > 0}

    parts  = []
    loaded = 0
    t0     = time.time()

    for i, f in enumerate(csv_files):
        q = quota.get(f, 0)
        if q == 0 or file_lengths.get(f, 0) == 0:
            continue

        try:
            df = pd.read_csv(f, usecols=ALL_FEATURES)
        except Exception as e:
            print(f"    Could not read {os.path.basename(f)}: {e}")
            continue

        missing = [c for c in ALL_FEATURES if c not in df.columns]
        if missing:
            print(f"    Missing columns {missing} in {os.path.basename(f)}")
            continue

        df = df.dropna(subset=ALL_FEATURES)
        df = df[df["packet_len"] > 0]
        if len(df) == 0:
            continue

        if len(df) > q:
            df = df.sample(n=q, random_state=seed)

        # Enforce types before log transform
        for c in CONTINUOUS_FEATURES:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype("float32")
        for c in BINARY_FEATURES:
            df[c] = (pd.to_numeric(df[c], errors="coerce")
                       .fillna(0).round().clip(0, 1).astype("int8"))

        # Apply log transforms
        df = apply_log_transforms(df, attack_name)

        df["_file_name"]  = os.path.basename(f)
        df["attack_name"] = attack_name
        parts.append(df)
        loaded += len(df)

        if (i + 1) % 10 == 0 or (i + 1) == len(csv_files):
            print(f"    Loaded {i+1}/{len(csv_files)} files | "
                  f"{loaded:,} packets | {elapsed(t0)}")

        if loaded >= max_packets:
            print(f"    Reached quota of {max_packets:,} — stopping early")
            break

    if not parts:
        return pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()

    print(f"    Final: {len(combined):,} packets from "
          f"{combined['_file_name'].nunique()} files")
    return combined


# TWO-STAGE GROUP SHUFFLE SPLIT
def three_way_split(df: pd.DataFrame,
                    test_ratio: float,
                    val_ratio_of_remainder: float,
                    seed: int) -> tuple:

    groups = df["_file_name"].values
    n_files = df["_file_name"].nunique()

    print(f"\n  Total packets : {len(df):,}")
    print(f"  Total files   : {n_files}")
    print(f"  Target split  : ~{(1-test_ratio-test_ratio)*100:.0f}% train / "
          f"~{test_ratio*100:.0f}% val / "
          f"~{test_ratio*100:.0f}% test")

    # Stage 1: split off test
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    trainval_idx, test_idx = next(gss1.split(np.arange(len(df)), groups=groups))

    trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
    test_df     = df.iloc[test_idx].reset_index(drop=True)

    # Stage 2: split trainval into train and val
    trainval_groups = trainval_df["_file_name"].values
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_ratio_of_remainder,
                              random_state=seed)
    train_idx, val_idx = next(
        gss2.split(np.arange(len(trainval_df)), groups=trainval_groups))

    train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
    val_df   = trainval_df.iloc[val_idx].reset_index(drop=True)

    return train_df, val_df, test_df


# MAIN
def main():
    total_start = time.time()

    print("=" * 70)
    print("NeuroStrike - Code 2 (FINAL): Preprocessing + Log Transforms")
    print("                                  + 70/15/15 Train/Val/Test Split")
    print("=" * 70)
    print(f"\nInput  : {INPUT_DIR}")
    print(f"Output : {OUTPUT_DIR}")
    print(f"\nLog transforms applied before saving:")
    print(f"  delta_time  → log1p(x * 1000)   [ALL attacks]")
    print(f"  payload_len → log1p(x)           [WILL + Invalid only]")
    print(f"\nInvert in Code 4 after generation:")
    print(f"  delta_time  → expm1(x) / 1000")
    print(f"  payload_len → expm1(x)           [WILL + Invalid only]")
    print(f"\nSplit strategy:")
    print(f"  GroupShuffleSplit on _file_name (no PCAP file crosses split boundary)")
    print(f"  Stage 1: carve out 15% test  → never seen by CTGAN")
    print(f"  Stage 2: split remainder 85% → 70% train / 15% val")
    print("=" * 70)

    # Step 1: Load all attack types
    print("\n[STEP 1/5] Loading and log-transforming packets...")
    print("-" * 70)

    all_dfs    = []
    norm_stats = {f: {"min": float("inf"), "max": float("-inf")}
                  for f in CONTINUOUS_FEATURES}
    attack_packet_counts = {}

    for attack_idx, attack_name in enumerate(ATTACK_TYPES):
        step_start = time.time()
        print(f"\n  [{attack_idx+1}/{len(ATTACK_TYPES)}] {attack_name}")

        transforms = ["delta_time: log1p(x*1000)"]
        if attack_name in PAYLOAD_LOG_ATTACKS:
            transforms.append("payload_len: log1p(x)")
        print(f"    Transforms: {transforms}")

        df = load_attack_type(attack_name, MAX_PACKETS_PER_ATTACK, RANDOM_SEED)

        if df.empty:
            print(f"    No data — skipping")
            attack_packet_counts[attack_name] = 0
            continue

        df["attack_label"] = attack_idx

        # Verify log transform worked — delta_time std should be ~1.0+
        dt_std = float(df["delta_time"].std())
        if dt_std < 0.5:
            print(f"    WARNING: delta_time std={dt_std:.4f} after transform")
            print(f"      Expected ~1.0+. Check that input data has valid timestamps.")
        else:
            print(f"    delta_time (log-space): "
                  f"mean={df['delta_time'].mean():.3f}  "
                  f"std={dt_std:.3f}  "
                  f"max={df['delta_time'].max():.3f}")

        if attack_name in PAYLOAD_LOG_ATTACKS:
            print(f"    payload_len (log-space): "
                  f"mean={df['payload_len'].mean():.3f}  "
                  f"std={df['payload_len'].std():.3f}  "
                  f"max={df['payload_len'].max():.3f}")

        for feat in CONTINUOUS_FEATURES:
            norm_stats[feat]["min"] = min(norm_stats[feat]["min"],
                                          float(df[feat].min()))
            norm_stats[feat]["max"] = max(norm_stats[feat]["max"],
                                          float(df[feat].max()))

        attack_packet_counts[attack_name] = len(df)
        all_dfs.append(df)

        print(f"    Done in {elapsed(step_start)}")
        del df
        gc.collect()

    if not all_dfs:
        raise RuntimeError("No packets loaded from any attack type.")

    print(f"\n  Combining all attack types...")
    t0 = time.time()
    full_df = pd.concat(all_dfs, ignore_index=True)
    del all_dfs
    gc.collect()
    print(f"  Total: {len(full_df):,} packets | "
          f"RAM: {full_df.memory_usage(deep=True).sum()/1e6:.0f} MB | "
          f"{elapsed(t0)}")

    # Step 2: Normalization stats
    print("\n[STEP 2/5] Normalization statistics (on log-transformed values)...")
    print("-" * 70)
    norm_ranges = {}
    for feat in CONTINUOUS_FEATURES:
        mn  = norm_stats[feat]["min"]
        mx  = norm_stats[feat]["max"]
        rng = max(mx - mn, 1e-9)
        norm_ranges[feat] = rng
        print(f"  {feat:<20} min={mn:.4f}  max={mx:.4f}  range={rng:.4f}")

    # Step 3: Three-way split
    print("\n[STEP 3/5] Three-way GroupShuffleSplit (file-level, no leakage)...")
    print("-" * 70)

    train_df, val_df, test_df = three_way_split(
        full_df,
        test_ratio=TEST_RATIO,
        val_ratio_of_remainder=VAL_RATIO,
        seed=RANDOM_SEED
    )
    del full_df
    gc.collect()

    # Verify no leakage
    verify_no_file_overlap(train_df, val_df, test_df)

    # Print actual split sizes
    total = len(train_df) + len(val_df) + len(test_df)
    print(f"\n  Actual split sizes:")
    print(f"    Train : {len(train_df):>10,}  ({len(train_df)/total*100:.1f}%)")
    print(f"    Val   : {len(val_df):>10,}  ({len(val_df)/total*100:.1f}%)")
    print(f"    Test  : {len(test_df):>10,}  ({len(test_df)/total*100:.1f}%)  ← held out")
    print(f"    Total : {total:>10,}")

    # Per-attack distribution across splits
    print(f"\n  Per-attack distribution:")
    print(f"  {'Attack Type':<45} {'Train':>8} {'Val':>8} {'Test':>8}")
    print(f"  {'-'*73}")
    for name in ATTACK_TYPES:
        tr = (train_df["attack_name"] == name).sum()
        vl = (val_df["attack_name"]   == name).sum()
        te = (test_df["attack_name"]  == name).sum()
        print(f"  {name:<45} {tr:>8,} {vl:>8,} {te:>8,}")

    # Step 4: Save outputs
    print("\n[STEP 4/5] Saving outputs...")
    print("-" * 70)

    drop_cols = ["_file_name"]

    # Train, val, test CSVs
    for split_name, split_df in [("train", train_df),
                                   ("val",   val_df),
                                   ("test",  test_df)]:
        t0   = time.time()
        out  = split_df.drop(columns=drop_cols, errors="ignore")
        path = OUTPUT_DIR / f"{split_name}_packets.csv"
        out.to_csv(path, index=False)
        note = "  ← CODE 6 LOADS FROM HERE" if split_name == "test" else ""
        print(f"  ✓ {split_name}_packets.csv  "
              f"({len(out):,} rows, {path.stat().st_size/1e6:.1f} MB)"
              f"{note} — {elapsed(t0)}")

    # Per-attack train CSVs for CTGAN
    print(f"\n  Saving per-attack train CSVs (CTGAN input)...")
    for attack_name in ATTACK_TYPES:
        t0   = time.time()
        mask = train_df["attack_name"] == attack_name
        sub  = train_df[mask][ALL_FEATURES + ["attack_label"]].copy()
        if len(sub) == 0:
            print(f"  No training rows for {attack_name}")
            continue
        path = PER_ATTACK_DIR / f"{attack_name}_train.csv"
        sub.to_csv(path, index=False)
        print(f"  ✓ {attack_name}_train.csv  "
              f"({len(sub):,} rows) — {elapsed(t0)}")

    del train_df, val_df, test_df
    gc.collect()

    # Step 5: Save normalization stats + split metadata 
    print("\n[STEP 5/5] Saving normalization stats...")
    print("-" * 70)

    stats_payload = {
        "continuous_features": {
            feat: {
                "min":   float(norm_stats[feat]["min"]),
                "max":   float(norm_stats[feat]["max"]),
                "range": float(norm_ranges[feat]),
                "note":  "values are in log-transformed space"
            }
            for feat in CONTINUOUS_FEATURES
        },
        "binary_features": {f: {} for f in BINARY_FEATURES},
        "feature_order":   ALL_FEATURES,
        "num_features":    NUM_FEATURES,
        "attack_labels":   {str(i): n for i, n in enumerate(ATTACK_TYPES)},
        "attack_packet_counts": attack_packet_counts,
        "log_transforms": {
            "delta_time": {
                "forward":  "log1p(x * 1000)",
                "inverse":  "expm1(x) / 1000",
                "attacks":  "ALL"
            },
            "payload_len": {
                "forward":  "log1p(x)",
                "inverse":  "expm1(x)",
                "attacks":  sorted(list(PAYLOAD_LOG_ATTACKS))
            }
        },
        "split_info": {
            "method":                 "Two-stage GroupShuffleSplit on _file_name",
            "test_ratio":             TEST_RATIO,
            "val_ratio_of_remainder": VAL_RATIO,
            "approximate_ratios":     "70% train / 15% val / 15% test",
            "leakage":                False,
            "note": (
                "test_packets.csv is held out for final evaluation only. "
                "Code 6 must load real reference data from test_packets.csv, "
                "NOT from Cleaned_Per_PCAP directly."
            )
        },
    }

    with open(OUTPUT_DIR / "normalization_stats.json", "w") as f:
        json.dump(stats_payload, f, indent=2)
    print(f"  ✓ normalization_stats.json saved")

    # Final summary 
    print("\n" + "=" * 70)
    print("CODE 2 COMPLETE")
    print("=" * 70)
    print(f"  Runtime : {elapsed(total_start)}")
    print(f"  Output  : {OUTPUT_DIR}")
    print(f"\n  Files created:")
    print(f"    train_packets.csv          ← CTGAN training data")
    print(f"    val_packets.csv            ← collapse monitoring")
    print(f"    test_packets.csv           ← Code 6 evaluation ONLY")
    print(f"    per_attack/*_train.csv     ← per-attack CTGAN input")
    print(f"    normalization_stats.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
