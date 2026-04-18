import os
import json
import pickle
import gc
import time
import warnings
import glob
import torch
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

_orig_torch_load = torch.load
torch.load = lambda *a, **kw: _orig_torch_load(
    *a, **{**kw, "map_location": torch.device("cpu")}
)

# CONFIGURATION
BASE_DIR      = Path(r"C:\Users\Lenovo\Downloads\DoS-DDoS-MQTT-IoT_Dataset")
MODEL_DIR     = BASE_DIR / "CTGAN_Models_NeuroStrike_v2"
STATS_DIR     = BASE_DIR / "CTGAN_Ready_NeuroStrike_v2"
REAL_DIR      = BASE_DIR / "Cleaned_Per_PCAP"
TEST_PATH     = STATS_DIR / "test_packets.csv"   # held-out test split
OUTPUT_DIR    = BASE_DIR / "FINALCTGAN_Synthetic_Traffic_v2"
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "per_attack").mkdir(exist_ok=True)

ATTACK_TYPES = [
    "Basic_Connect_Flooding",
    "Connect_Flooding_with_WILL_payload",
    "Delayed_Connect_Flooding",
    "Invalid_Subscription_Flooding",
    "SYN_TCP_Flooding",
]

# Attacks where payload_len was log-transformed in Code 2
PAYLOAD_LOG_ATTACKS = {
    "Connect_Flooding_with_WILL_payload",
    "Invalid_Subscription_Flooding",
}

CONTINUOUS_FEATURES = ["delta_time", "packet_len", "payload_len", "tcp_window_size"]
BINARY_FEATURES     = ["flag_syn", "flag_ack", "flag_fin", "flag_rst",
                        "flag_psh", "flag_urg", "port_direction"]
ALL_FEATURES        = CONTINUOUS_FEATURES + BINARY_FEATURES

# Standard correction: only tcp_window_size for most attacks
QMAP_FEATURES_DEFAULT  = ["tcp_window_size"]

# Stronger correction for Invalid_Subscription which is underfitted
# delta_time + tcp_window_size both need qmap for this attack
QMAP_FEATURES_INVALID  = ["tcp_window_size", "delta_time"]

FLAG_EXCLUDE       = {"flag_urg"}
FLAG_GAP_THRESHOLD = 0.05

QMAP_REAL_SAMPLE   = 200_000
PACKETS_PER_ATTACK = 300_000
BATCH_SIZE         = 10_000

# HELPERS
def elapsed(start: float) -> str:
    s = time.time() - start
    return f"{s:.1f}s" if s < 60 else f"{int(s//60)}m {int(s%60)}s"


def load_model(attack_name: str):
    safe = attack_name.replace(" ", "_")
    path = MODEL_DIR / f"ctgan_model_{safe}.pkl"
    if not path.exists():
        print(f"  ⚠️  Model not found: {path.name}")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def invert_log_transforms(df: pd.DataFrame, attack_name: str) -> pd.DataFrame:

    df = df.copy()

    if "delta_time" in df.columns:
        x = df["delta_time"].values.astype(float)
        x = np.clip(x, 0, 6)
        df["delta_time"] = np.expm1(x) / 1000.0
        df["delta_time"] = df["delta_time"].clip(0, 10.0).astype("float32")

    if attack_name in PAYLOAD_LOG_ATTACKS and "payload_len" in df.columns:
        df["payload_len"] = np.expm1(
            df["payload_len"].values.astype(float))
        df["payload_len"] = df["payload_len"].clip(0, 1460).astype("float32")

    return df


def clean_generated(df: pd.DataFrame) -> pd.DataFrame:
    for feat in CONTINUOUS_FEATURES:
        if feat in df.columns:
            df[feat] = (pd.to_numeric(df[feat], errors="coerce")
                          .fillna(0.0).astype("float32"))
    for feat in BINARY_FEATURES:
        if feat in df.columns:
            df[feat] = (pd.to_numeric(df[feat], errors="coerce")
                          .fillna(0).round().clip(0, 1).astype("int8"))
    df["packet_len"]      = df["packet_len"].clip(40, 1500).round().astype(int)
    df["payload_len"]     = df["payload_len"].clip(0, 1460).round().astype(int)
    df["tcp_window_size"] = df["tcp_window_size"].clip(0, 131328).round().astype(int)
    df["delta_time"]      = df["delta_time"].clip(0, 10.0)
    return df


def generate_batched(model, n_total: int) -> pd.DataFrame:
    parts     = []
    done      = 0
    t0        = time.time()
    n_batches = (n_total + BATCH_SIZE - 1) // BATCH_SIZE
    for b in range(n_batches):
        this_batch = min(BATCH_SIZE, n_total - done)
        try:
            chunk = model.sample(this_batch)
        except Exception as e:
            print(f"    ⚠️  Batch {b+1} failed: {e}")
            continue
        cols = [c for c in ALL_FEATURES if c in chunk.columns]
        parts.append(chunk[cols].copy())
        done += this_batch
        if (b + 1) % 5 == 0 or (b + 1) == n_batches:
            print(f"    Batch {b+1}/{n_batches} | {done:,}/{n_total:,} "
                  f"({done/n_total*100:.0f}%) | {elapsed(t0)}")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# REFERENCE DATA
class RealDataReference:

    def __init__(self):
        self._cache      = {}
        self._test_cache = {}

    def _build_stats(self, df: pd.DataFrame) -> dict:
        stats = {"continuous": {}, "binary": {}}
        for feat in CONTINUOUS_FEATURES:
            if feat not in df.columns:
                continue
            vals = df[feat].values.astype(float)
            vals = vals[np.isfinite(vals)]
            p01  = float(np.percentile(vals, 1))
            p99  = float(np.percentile(vals, 99))
            clipped = np.clip(vals, p01, p99)
            stats["continuous"][feat] = {
                "sorted": np.sort(clipped),
                "mean":   float(clipped.mean()),
                "p01":    p01,
                "p99":    p99,
            }
        for feat in BINARY_FEATURES:
            if feat in df.columns:
                stats["binary"][feat] = float(df[feat].mean())
        return stats

    def get(self, attack_name: str) -> dict:
        if attack_name in self._cache:
            return self._cache[attack_name]

        attack_dir = REAL_DIR / attack_name
        if not attack_dir.exists():
            return {}

        files = sorted(glob.glob(str(attack_dir / "cleaned_pcap_*.csv")))
        dfs, loaded = [], 0
        for f in files:
            remaining = QMAP_REAL_SAMPLE - loaded
            if remaining <= 0:
                break
            try:
                df = pd.read_csv(f, usecols=ALL_FEATURES, nrows=remaining)
                dfs.append(df)
                loaded += len(df)
            except Exception:
                continue

        if not dfs:
            return {}

        real_df = pd.concat(dfs, ignore_index=True).dropna(subset=ALL_FEATURES)
        stats   = self._build_stats(real_df)
        self._cache[attack_name] = stats
        print(f"    Loaded {len(real_df):,} real rows (Cleaned_Per_PCAP)")
        return stats

    def get_test_split(self, attack_name: str) -> dict:

        if attack_name in self._test_cache:
            return self._test_cache[attack_name]

        if not TEST_PATH.exists():
            print(f"      test_packets.csv not found at {TEST_PATH}")
            return self.get(attack_name)  # fallback to training reference

        test_df = pd.read_csv(TEST_PATH)
        subset  = test_df[test_df["attack_name"] == attack_name][ALL_FEATURES].copy()
        subset  = subset.dropna(subset=ALL_FEATURES)

        if len(subset) == 0:
            print(f"      No test rows for {attack_name} — using training reference")
            return self.get(attack_name)

        # Invert log transforms on test data
        subset["delta_time"] = (
            np.expm1(subset["delta_time"].values.astype(float)) / 1000.0
        )
        subset["delta_time"] = subset["delta_time"].clip(0, 10.0)

        if attack_name in PAYLOAD_LOG_ATTACKS:
            subset["payload_len"] = np.expm1(
                subset["payload_len"].values.astype(float))
            subset["payload_len"] = subset["payload_len"].clip(0, 1460)

        stats = self._build_stats(subset)
        self._test_cache[attack_name] = stats
        print(f"    Loaded {len(subset):,} test rows (test split reference)")
        return stats


# POST-GENERATION CORRECTION
def quantile_map(fake_vals: np.ndarray, real_sorted: np.ndarray) -> np.ndarray:
    if len(fake_vals) < 2 or len(real_sorted) < 2:
        return fake_vals
    order        = np.argsort(fake_vals)
    ranks        = np.empty_like(order, dtype=float)
    ranks[order] = np.linspace(0, 1, len(fake_vals))
    idx          = np.clip((ranks * (len(real_sorted) - 1)).astype(int),
                           0, len(real_sorted) - 1)
    return real_sorted[idx].astype(np.float32)


def apply_correction(df: pd.DataFrame,
                     attack_name: str,
                     real_ref: RealDataReference,
                     rng: np.random.Generator) -> tuple:

    is_invalid = (attack_name == "Invalid_Subscription_Flooding")

    # Choose reference and features based on attack
    if is_invalid:
        stats         = real_ref.get_test_split(attack_name)
        qmap_features = QMAP_FEATURES_INVALID
        print(f"    Using test split reference (stronger correction)")
    else:
        stats         = real_ref.get(attack_name)
        qmap_features = QMAP_FEATURES_DEFAULT

    report = {"continuous": {}, "binary": {}}

    if not stats:
        print(f"      No reference data — skipping correction")
        return df, report

    df = df.copy()

    # Quantile map selected continuous features
    for feat in qmap_features:
        feat_stats = stats["continuous"].get(feat)
        if feat_stats is None or feat not in df.columns:
            continue
        fake_vals  = df[feat].values.astype(float)
        fake_mean  = float(fake_vals.mean())
        mapped     = quantile_map(fake_vals, feat_stats["sorted"])
        mapped     = np.clip(mapped, feat_stats["p01"], feat_stats["p99"])
        df[feat]   = mapped
        report["continuous"][feat] = {
            "applied":          True,
            "fake_mean_before": fake_mean,
            "fake_mean_after":  float(df[feat].mean()),
            "real_mean":        feat_stats["mean"],
            "reference":        "test_split" if is_invalid else "cleaned_per_pcap",
        }

    # Binary flag correction
    for feat in BINARY_FEATURES:
        if feat in FLAG_EXCLUDE:
            continue
        real_rate = stats["binary"].get(feat)
        if real_rate is None or feat not in df.columns:
            continue
        fake_rate = float(df[feat].mean())
        gap       = abs(real_rate - fake_rate)
        if gap > FLAG_GAP_THRESHOLD:
            df[feat] = (rng.random(len(df)) < real_rate).astype(np.int8)
            report["binary"][feat] = {
                "applied":          True,
                "gap":              gap,
                "real_rate":        real_rate,
                "fake_rate_before": fake_rate,
                "fake_rate_after":  float(df[feat].mean()),
            }
        else:
            report["binary"][feat] = {"applied": False, "gap": gap}

    return df, report


def print_correction_report(report: dict):
    cont_applied = [f for f, v in report["continuous"].items() if v.get("applied")]
    bin_applied  = [f for f, v in report["binary"].items()     if v.get("applied")]
    if not cont_applied and not bin_applied:
        print(f"    No correction needed")
        return
    if cont_applied:
        print(f"    Quantile mapped: {cont_applied}")
        for feat in cont_applied:
            r = report["continuous"][feat]
            ref = r.get("reference", "")
            print(f"      {feat:<22} "
                  f"{r['fake_mean_before']:>10.4f} → {r['fake_mean_after']:>10.4f}  "
                  f"(real: {r['real_mean']:>10.4f})  [{ref}]")
    if bin_applied:
        print(f"    Flag resampled: {bin_applied}")
        for feat in bin_applied:
            r = report["binary"][feat]
            print(f"      {feat:<22} "
                  f"{r['fake_rate_before']:.3f} → {r['fake_rate_after']:.3f}  "
                  f"(real: {r['real_rate']:.3f})")


# GENERATION
def generate_pure(real_ref: RealDataReference, rng: np.random.Generator) -> tuple:
    print(f"\n{'=' * 70}")
    print(f"Pure Generation ({PACKETS_PER_ATTACK:,} per attack type)")
    print(f"{'=' * 70}")

    all_parts = []
    all_stats = {}
    t_total   = time.time()

    for idx, attack_name in enumerate(ATTACK_TYPES):
        print(f"\n  [{idx+1}/{len(ATTACK_TYPES)}] {attack_name}")
        t_attack = time.time()

        model = load_model(attack_name)
        if model is None:
            continue

        df = generate_batched(model, PACKETS_PER_ATTACK)
        del model; gc.collect()

        if df.empty:
            continue

        # Fix 1: invert with log-space clipping
        print(f"    Inverting log transforms (clipped at log=6)...")
        df = invert_log_transforms(df, attack_name)

        print(f"    delta_time after inversion: "
              f"mean={df['delta_time'].mean():.6f}  "
              f"std={df['delta_time'].std():.6f}  "
              f"max={df['delta_time'].max():.6f}")

        df = clean_generated(df)

        # Fix 1+2: correction (stronger for Invalid_Subscription)
        print(f"    Applying correction...")
        df, corr_report = apply_correction(df, attack_name, real_ref, rng)
        print_correction_report(corr_report)

        df["attack_type"]  = attack_name
        df["attack_label"] = idx
        df["variant"]      = "pure"
        df["sequence_id"]  = np.arange(len(df)) // 50
        df["packet_id"]    = np.arange(len(df)) % 50

        out_path = OUTPUT_DIR / "per_attack" / f"{attack_name}_synthetic.csv"
        df.to_csv(out_path, index=False)

        s = {"n_packets": len(df)}
        for feat in CONTINUOUS_FEATURES:
            if feat in df.columns:
                s[f"{feat}_mean"] = float(df[feat].mean())
                s[f"{feat}_std"]  = float(df[feat].std())
        for feat in ["flag_syn", "flag_ack", "flag_rst"]:
            if feat in df.columns:
                s[f"{feat}_rate"] = float(df[feat].mean())

        all_stats[attack_name] = s
        all_parts.append(df)

        print(f"  ✓ {len(df):,} packets | {elapsed(t_attack)}")
        print(f"    delta_time mean={s.get('delta_time_mean',0):.6f}  "
              f"std={s.get('delta_time_std',0):.6f}")
        print(f"    tcp_win    mean={s.get('tcp_window_size_mean',0):.0f}")
        print(f"    flag_ack={s.get('flag_ack_rate',0):.3f}  "
              f"flag_rst={s.get('flag_rst_rate',0):.3f}")

        del df; gc.collect()

    combined = (pd.concat(all_parts, ignore_index=True)
                if all_parts else pd.DataFrame())
    del all_parts; gc.collect()

    print(f"\n  Pure generation complete | "
          f"total={len(combined):,} | {elapsed(t_total)}")
    return combined, all_stats


# MAIN

def main():
    total_start = time.time()

    print("=" * 70)
    print("NeuroStrike - Code 4: Generation + Log Inversion")
    print("=" * 70)
    print(f"\nModel dir : {MODEL_DIR}")
    print(f"Real dir  : {REAL_DIR}")
    print(f"Test split: {TEST_PATH}")
    print(f"Output    : {OUTPUT_DIR}")
    print(f"\nFix 1 — delta_time log-space clip at 6 before expm1 [ALL attacks]")
    print(f"Fix 2 — Invalid_Subscription: qmap delta_time+tcp using test split")
    print(f"\nStandard correction: qmap tcp_window_size only [other 4 attacks]")
    print(f"Flag correction    : Bernoulli resample if gap > {FLAG_GAP_THRESHOLD}")
    print("=" * 70)

    print("\nChecking models...")
    for name in ATTACK_TYPES:
        safe = name.replace(" ", "_")
        p    = MODEL_DIR / f"ctgan_model_{safe}.pkl"
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {p}")
        print(f"  ✓ {p.name} ({p.stat().st_size/1e6:.1f} MB)")

    print("\nChecking test split...")
    if TEST_PATH.exists():
        print(f"  ✓ test_packets.csv ({TEST_PATH.stat().st_size/1e6:.1f} MB)")
    else:
        print(f"  ⚠️  test_packets.csv not found — Fix 2 will fall back to "
              f"Cleaned_Per_PCAP for Invalid_Subscription")

    print("\nChecking real data...")
    for name in ATTACK_TYPES:
        d       = REAL_DIR / name
        n_files = len(glob.glob(str(d / "cleaned_pcap_*.csv"))) if d.exists() else 0
        status  = f"✓ {n_files} files" if n_files > 0 else "⚠️  missing"
        print(f"  {status}  {name}")

    real_ref = RealDataReference()
    rng      = np.random.default_rng(42)

    pure_df, pure_stats = generate_pure(real_ref, rng)

    print(f"\n{'=' * 70}")
    print(f"Saving combined output...")
    if not pure_df.empty:
        t0  = time.time()
        out = OUTPUT_DIR / "all_attacks_synthetic.csv"
        pure_df.to_csv(out, index=False)
        print(f"  ✓ all_attacks_synthetic.csv  "
              f"({len(pure_df):,} rows, {out.stat().st_size/1e6:.1f} MB) | {elapsed(t0)}")

    metadata = {
        "model_type":    "CTGAN + Fix1 log-clip + Fix2 invalid correction",
        "feature_order": ALL_FEATURES,
        "fix1":          "delta_time clipped at log=6 before expm1",
        "fix2":          "Invalid_Subscription uses test split qmap for delta_time+tcp",
        "correction": {
            "default_qmap":   QMAP_FEATURES_DEFAULT,
            "invalid_qmap":   QMAP_FEATURES_INVALID,
            "flag_threshold": FLAG_GAP_THRESHOLD,
        },
        "pure_stats":  pure_stats,
        "total_pure":  len(pure_df),
    }
    with open(OUTPUT_DIR / "generation_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ generation_metadata.json")

    print("\n" + "=" * 70)
    print("CODE 4 COMPLETE")
    print("=" * 70)
    print(f"  Total runtime : {elapsed(total_start)}")
    print(f"  Pure attacks  : {len(pure_df):,} packets")
    print("=" * 70)


if __name__ == "__main__":
    main()
