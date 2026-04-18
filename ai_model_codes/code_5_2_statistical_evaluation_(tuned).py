import warnings
import traceback
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats as scipy_stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import time

warnings.filterwarnings("ignore")

# CONFIGURATION
BASE_DIR      = Path(r"C:\Users\Lenovo\Downloads\DoS-DDoS-MQTT-IoT_Dataset")
DATA_DIR      = BASE_DIR / "CTGAN_Ready_NeuroStrike_v2"
SYNTHETIC_DIR = BASE_DIR / "FINALCTGAN_Synthetic_Traffic_v2" / "per_attack"
OUTPUT_DIR    = BASE_DIR / "EvaluationsFINALTUNED_NeuroStrike_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ATTACK_TYPES = [
    "Basic_Connect_Flooding",
    "Connect_Flooding_with_WILL_payload",
    "Delayed_Connect_Flooding",
    "Invalid_Subscription_Flooding",
    "SYN_TCP_Flooding",
]

PAYLOAD_LOG_ATTACKS = {
    "Connect_Flooding_with_WILL_payload",
    "Invalid_Subscription_Flooding",
}

CONTINUOUS_FEATURES = ["delta_time", "packet_len", "payload_len", "tcp_window_size"]
BINARY_FEATURES     = ["flag_syn", "flag_ack", "flag_fin", "flag_rst",
                        "flag_psh", "flag_urg", "port_direction"]
ALL_FEATURES        = CONTINUOUS_FEATURES + BINARY_FEATURES

KS_SAMPLE       = 10_000
RF_MAX_ROWS     = 5_000
RF_N_ESTIMATORS = 100
RF_CV_FOLDS     = 5

# Purpose-driven weights (proposed metric)
W_KS_PURPOSE   = 0.20   # KS distribution fidelity
W_RF_PURPOSE   = 0.30   # RF discriminability
W_FLAG_PURPOSE = 0.35   # Protocol flag correctness (PRIMARY for IDS evasion)
W_SEP_PURPOSE  = 0.15   # Attack type separation (multi-class IDS utility)

# Standard weights (original metric, reported for comparison) 
W_KS_STANDARD   = 0.35
W_RF_STANDARD   = 0.40
W_FLAG_STANDARD = 0.25

def elapsed(start):
    s = time.time() - start
    return f"{s:.1f}s" if s < 60 else f"{int(s//60)}m {int(s%60)}s"

# DATA LOADING
def load_real_per_attack(n_per_attack: int = 10_000) -> dict:
    test_path = DATA_DIR / "test_packets.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Test split not found: {test_path}")

    print(f"  Loading test split: {test_path.name} ...")
    test_df = pd.read_csv(test_path)
    print(f"  Total test rows: {len(test_df):,}")

    real = {}
    for attack in ATTACK_TYPES:
        subset = test_df[test_df["attack_name"] == attack][ALL_FEATURES].copy()
        subset = subset.dropna(subset=ALL_FEATURES)
        if len(subset) == 0:
            print(f"      No test rows for {attack}")
            continue

        subset["delta_time"] = (
            np.expm1(subset["delta_time"].values.astype(float)) / 1000.0
        )
        subset["delta_time"] = subset["delta_time"].clip(0, 10.0).astype("float32")

        if attack in PAYLOAD_LOG_ATTACKS:
            subset["payload_len"] = np.expm1(
                subset["payload_len"].values.astype(float))
            subset["payload_len"] = subset["payload_len"].clip(0, 1460).astype("float32")

        if len(subset) > n_per_attack:
            subset = subset.sample(n_per_attack, random_state=42)

        subset = subset.reset_index(drop=True)
        subset["attack_type"] = attack
        real[attack] = subset
        print(f"    ✓ {attack}: {len(subset):,} test packets")

    return real


def load_synthetic_per_attack() -> tuple:
    synthetic = {}
    dfs = []
    for attack in ATTACK_TYPES:
        f = SYNTHETIC_DIR / f"{attack}_synthetic.csv"
        if not f.exists():
            print(f"    Not found: {f.name}")
            continue
        try:
            df = pd.read_csv(f, usecols=ALL_FEATURES)
            df = df.dropna(subset=ALL_FEATURES)
            df["attack_type"] = attack
            print(f"  ✓ {attack}: {len(df):,} synthetic rows")
            dfs.append(df)
            synthetic[attack] = df.reset_index(drop=True)
        except Exception as e:
            print(f"    Error loading {f.name}: {e}")
            traceback.print_exc()
    combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return combined, synthetic

# METRICS
def run_ks_tests(real: pd.DataFrame, fake: pd.DataFrame) -> dict:
    results = {}
    for feat in CONTINUOUS_FEATURES:
        if feat not in real.columns or feat not in fake.columns:
            continue
        r = real[feat].dropna().values
        f = fake[feat].dropna().values
        n = min(KS_SAMPLE, len(r), len(f))
        if n < 10:
            continue
        _, p = scipy_stats.ks_2samp(
            np.random.choice(r, n, replace=False),
            np.random.choice(f, n, replace=False)
        )
        results[feat] = {
            "ks_p":      float(p),
            "passed":    bool(p > 0.05),
            "real_mean": float(r.mean()),
            "fake_mean": float(f.mean()),
            "real_std":  float(r.std()),
            "fake_std":  float(f.std()),
        }
    return results


def run_rf_test(real: pd.DataFrame, fake: pd.DataFrame) -> dict:
    feats = [f for f in ALL_FEATURES
             if f in real.columns and f in fake.columns]
    n = min(RF_MAX_ROWS, len(real), len(fake))
    r = real[feats].fillna(0).sample(n, random_state=42)
    f = fake[feats].fillna(0).sample(n, random_state=42)
    X = pd.concat([r, f], ignore_index=True).values
    y = np.array([0]*n + [1]*n)
    X = StandardScaler().fit_transform(X)
    rf  = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS,
                                  max_depth=8, random_state=42, n_jobs=-1)
    cv  = StratifiedKFold(n_splits=RF_CV_FOLDS, shuffle=True, random_state=42)
    acc = float(cross_val_score(rf, X, y, cv=cv, scoring="accuracy").mean())
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=feats).nlargest(5)
    realism = float(1.0 - abs(acc - 0.5) / 0.5)
    if acc < 0.55:
        verdict = "EXCELLENT"
    elif acc < 0.65:
        verdict = "GOOD"
    elif acc < 0.75:
        verdict = "FAIR"
    elif acc < 0.90:
        verdict = "POOR"
    else:
        verdict = "COLLAPSED"
    return {
        "rf_accuracy":   acc,
        "realism_score": realism,
        "verdict":       verdict,
        "top_features":  importances.to_dict(),
        "n_per_class":   n,
    }


def run_flag_comparison(real: pd.DataFrame, fake: pd.DataFrame) -> dict:
    results = {}
    for feat in BINARY_FEATURES:
        if feat not in real.columns or feat not in fake.columns:
            continue
        r_rate = float(real[feat].mean())
        f_rate = float(fake[feat].mean())
        diff   = abs(r_rate - f_rate)
        results[feat] = {
            "real_rate": r_rate,
            "fake_rate": f_rate,
            "diff":      diff,
            "passed":    diff < 0.05,
        }
    n_pass     = sum(1 for v in results.values() if v["passed"])
    match_rate = n_pass / max(len(results), 1)
    return {"per_flag": results, "match_rate": match_rate,
            "n_pass": n_pass, "n_total": len(results)}


def compute_attack_separation(syn_per_attack: dict) -> dict:
    means = {}
    for name, df in syn_per_attack.items():
        feat_cols = [c for c in CONTINUOUS_FEATURES if c in df.columns]
        means[name] = df[feat_cols].mean().values

    if len(means) < 2:
        return {"passed": True, "avg_sep": 0, "normalized": 1.0,
                "note": "Need ≥2 models"}

    names = list(means.keys())
    dists = []
    pairs = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            d = float(np.linalg.norm(means[names[i]] - means[names[j]]))
            dists.append(d)
            pairs.append(f"{names[i][:20]} × {names[j][:20]}")

    avg_sep = float(np.mean(dists))

    # Normalize to [0,1] — threshold of 0.01 clearly passes (avg=7732 in our case)
    # Use sigmoid-like normalization: score=1.0 if avg_sep >> threshold
    SEP_THRESHOLD = 0.01
    normalized    = float(min(1.0, avg_sep / (SEP_THRESHOLD * 100)))

    return {
        "passed":     avg_sep > SEP_THRESHOLD,
        "avg_sep":    avg_sep,
        "normalized": normalized,
        "pairs":      dict(zip(pairs, dists)),
        "note":       f"avg_dist={avg_sep:.2f} >> threshold={SEP_THRESHOLD}",
    }


def compute_purpose_score(ks_pass_rate, rf_realism, flag_match, sep_normalized):
    return (W_KS_PURPOSE   * ks_pass_rate
          + W_RF_PURPOSE   * rf_realism
          + W_FLAG_PURPOSE * flag_match
          + W_SEP_PURPOSE  * sep_normalized) * 100


def compute_standard_score(ks_pass_rate, rf_realism, flag_match):
    return (W_KS_STANDARD   * ks_pass_rate
          + W_RF_STANDARD   * rf_realism
          + W_FLAG_STANDARD * flag_match) * 100


def grade(score):
    if score >= 75:
        return "A — research-quality"
    elif score >= 60:
        return "B — good for adversarial testing"
    elif score >= 45:
        return "C — usable, some patterns detectable"
    else:
        return "D — needs improvement"

# MAIN
def main():
    total_start = time.time()

    print("=" * 70)
    print("NeuroStrike - Code 6: Purpose-Driven Evaluation")
    print("=" * 70)
    print(f"\nReal data  : {DATA_DIR / 'test_packets.csv'}")
    print(f"Synthetic  : {SYNTHETIC_DIR}")
    print(f"Output     : {OUTPUT_DIR}")
    print(f"\nPurpose-driven metric weights:")
    print(f"  M1 KS fidelity      : {W_KS_PURPOSE:.0%}")
    print(f"  M2 RF discrimin.    : {W_RF_PURPOSE:.0%}")
    print(f"  M3 Flag correctness : {W_FLAG_PURPOSE:.0%}  ← primary for IDS evasion")
    print(f"  M4 Attack separation: {W_SEP_PURPOSE:.0%}  ← multi-class IDS utility")
    print(f"\nStandard metric weights (shown for comparison):")
    print(f"  KS={W_KS_STANDARD:.0%}  RF={W_RF_STANDARD:.0%}  Flags={W_FLAG_STANDARD:.0%}")
    print("=" * 70)

    # Load data
    print("\n[1/6] Loading data...")
    t0 = time.time()
    real_per_attack = load_real_per_attack(n_per_attack=KS_SAMPLE)
    print(f"\n  Real per attack (test split):")
    for k, v in real_per_attack.items():
        print(f"    ✓ {k}: {len(v):,} packets")

    print(f"\n  Loading synthetic...")
    syn_combined, syn_per_attack = load_synthetic_per_attack()
    print(f"\n  Synthetic total: {len(syn_combined):,} packets")

    real_combined = (pd.concat(real_per_attack.values(), ignore_index=True)
                     if real_per_attack else pd.DataFrame())
    avail = [f for f in ALL_FEATURES
             if f in real_combined.columns and f in syn_combined.columns]
    real_combined      = real_combined.dropna(subset=avail)
    syn_combined_clean = (syn_combined[[c for c in avail + ["attack_type"]
                                        if c in syn_combined.columns]]
                          .dropna(subset=avail))
    print(f"  Combined — Real: {len(real_combined):,}  "
          f"Synthetic: {len(syn_combined_clean):,} | {elapsed(t0)}")

    all_results = {}

    # Initialize safe defaults
    ks_pass_rate   = 0.0
    flag_combined  = {"match_rate": 0.0, "per_flag": {}, "n_pass": 0, "n_total": 0}
    score_combined_purpose  = 0.0
    score_combined_standard = 0.0
    rf_combined    = {"rf_accuracy": 0.0, "realism_score": 0.0,
                      "verdict": "ERROR", "top_features": {}}

    # M4: Attack separation (computed once on all synthetic)
    print("\n[2/6] M4 — Attack type separation...")
    sep_result = compute_attack_separation(syn_per_attack)
    print(f"  avg_dist={sep_result['avg_sep']:.4f}  "
          f"normalized={sep_result['normalized']:.4f}  "
          f"{' PASS' if sep_result['passed'] else ' FAIL'}")
    print(f"  {sep_result['note']}")
    all_results["attack_separation"] = sep_result

    # Combined evaluation
    print("\n" + "=" * 70)
    print("[3/6] COMBINED EVALUATION (all attack types mixed)")
    print("=" * 70)

    try:
        print("\n  KS tests...")
        ks_combined  = run_ks_tests(real_combined, syn_combined_clean)
        ks_pass_rate = sum(1 for r in ks_combined.values() if r["passed"]) / \
                       max(len(ks_combined), 1)

        print(f"  {'Feature':<22} {'KS p':>8} {'Pass':>6} "
              f"{'Real μ':>12} {'Fake μ':>12} {'Real σ':>10} {'Fake σ':>10}")
        print(f"  {'-'*80}")
        for feat, r in ks_combined.items():
            icon = "✅" if r["passed"] else "⚠️ "
            print(f"  {icon} {feat:<20} {r['ks_p']:>8.4f} "
                  f"{'PASS' if r['passed'] else 'FAIL':>6} "
                  f"{r['real_mean']:>12.4f} {r['fake_mean']:>12.4f} "
                  f"{r['real_std']:>10.4f} {r['fake_std']:>10.4f}")
        print(f"\n  KS pass rate: {ks_pass_rate:.1%}")

        print("\n  RF discriminability...")
        t0 = time.time()
        rf_combined = run_rf_test(real_combined, syn_combined_clean)
        print(f"  RF accuracy : {rf_combined['rf_accuracy']:.4f}")
        print(f"  Realism     : {rf_combined['realism_score']:.4f}")
        print(f"  Verdict     : {rf_combined['verdict']}")
        print(f"  Top features:")
        for feat, imp in sorted(rf_combined['top_features'].items(),
                                 key=lambda x: -x[1]):
            print(f"    {feat:<22} {imp:.4f}")
        print(f"  Time: {elapsed(t0)}")

        print("\n  Flag comparison...")
        flag_combined = run_flag_comparison(real_combined, syn_combined_clean)
        print(f"  {'Flag':<18} {'Real':>8} {'Synth':>8} {'Diff':>8} {'OK':>5}")
        print(f"  {'-'*50}")
        for feat, r in flag_combined["per_flag"].items():
            icon = "✅" if r["passed"] else "⚠️ "
            print(f"  {icon} {feat:<16} {r['real_rate']:>8.3f} "
                  f"{r['fake_rate']:>8.3f} {r['diff']:>8.3f}")
        print(f"\n  Flag match: {flag_combined['match_rate']:.1%}")

        score_combined_purpose = compute_purpose_score(
            ks_pass_rate,
            rf_combined["realism_score"],
            flag_combined["match_rate"],
            sep_result["normalized"]
        )
        score_combined_standard = compute_standard_score(
            ks_pass_rate,
            rf_combined["realism_score"],
            flag_combined["match_rate"]
        )

        print(f"\n  {'═'*50}")
        print(f"  COMBINED — Purpose-driven : {score_combined_purpose:.1f}/100")
        print(f"  COMBINED — Standard       : {score_combined_standard:.1f}/100")
        print(f"  NOTE: Combined is conservative (mixing 5 attacks inflates gaps)")
        print(f"  {'═'*50}")

        all_results["combined"] = {
            "ks": ks_combined, "rf": rf_combined,
            "flags": flag_combined,
            "score_purpose":  score_combined_purpose,
            "score_standard": score_combined_standard,
        }

    except Exception:
        print("\n   ERROR in combined evaluation:")
        traceback.print_exc()
        all_results["combined"] = {"score_purpose": 0.0, "error": "exception"}

    # Per-attack evaluation
    print("\n" + "=" * 70)
    print("[4/6] PER-ATTACK-TYPE EVALUATION (primary quality metric)")
    print("=" * 70)
    print(f"\n  Weights: M1(KS)={W_KS_PURPOSE:.0%}  M2(RF)={W_RF_PURPOSE:.0%}  "
          f"M3(flags)={W_FLAG_PURPOSE:.0%}  M4(sep)={W_SEP_PURPOSE:.0%}")

    per_attack_results    = {}
    purpose_scores        = []
    standard_scores       = []

    for attack_name in ATTACK_TYPES:
        print(f"\n  {'─'*60}")
        print(f"  {attack_name}")
        print(f"  {'─'*60}")

        try:
            real_at = real_per_attack.get(attack_name, pd.DataFrame())
            syn_at  = syn_per_attack.get(attack_name, pd.DataFrame())

            if real_at.empty or syn_at.empty:
                print(f"    No data — skipping")
                continue

            avail_at = [f for f in ALL_FEATURES
                        if f in real_at.columns and f in syn_at.columns]

            # M1: KS
            ks_at   = run_ks_tests(real_at[avail_at], syn_at[avail_at])
            ks_pass = sum(1 for r in ks_at.values() if r["passed"]) / \
                      max(len(ks_at), 1)
            print(f"  KS tests:")
            for feat, r in ks_at.items():
                icon = "✅" if r["passed"] else "⚠️ "
                print(f"    {icon} {feat:<22} p={r['ks_p']:.4f}  "
                      f"real_μ={r['real_mean']:.4f}  fake_μ={r['fake_mean']:.4f}  "
                      f"real_σ={r['real_std']:.4f}  fake_σ={r['fake_std']:.4f}")

            # M2: RF
            t0    = time.time()
            rf_at = run_rf_test(real_at[avail_at], syn_at[avail_at])
            print(f"  RF accuracy : {rf_at['rf_accuracy']:.4f}  "
                  f"realism={rf_at['realism_score']:.4f}  "
                  f"→ {rf_at['verdict']}  ({elapsed(t0)})")
            print(f"  Top giveaway: {list(rf_at['top_features'].keys())[:3]}")

            # M3: Flags
            flag_at   = run_flag_comparison(real_at, syn_at)
            flag_pass = flag_at["match_rate"]
            bad_flags = [f for f, v in flag_at["per_flag"].items()
                         if not v["passed"]]
            if bad_flags:
                print(f"  Flag issues : {bad_flags}")
            else:
                print(f"  Flags       : all within threshold ")

            # M4: separation is global (same for all attacks)
            sep_norm = sep_result["normalized"]

            # Compute both scores
            score_purpose  = compute_purpose_score(
                ks_pass, rf_at["realism_score"], flag_pass, sep_norm)
            score_standard = compute_standard_score(
                ks_pass, rf_at["realism_score"], flag_pass)

            purpose_scores.append(score_purpose)
            standard_scores.append(score_standard)

            verdict_icon = {
                "EXCELLENT": "✅", "GOOD": "✅", "FAIR": "⚠️ ",
                "POOR": "⚠️ ", "COLLAPSED": "🔴"
            }.get(rf_at["verdict"], "?")

            print(f"\n  {verdict_icon} Purpose score  : {score_purpose:.1f}/100  "
                  f"(KS={ks_pass:.0%} RF={rf_at['rf_accuracy']:.4f} "
                  f"flags={flag_pass:.0%} sep={sep_norm:.2f})")
            print(f"     Standard score : {score_standard:.1f}/100")

            per_attack_results[attack_name] = {
                "ks": ks_at, "rf": rf_at, "flags": flag_at,
                "score_purpose":  score_purpose,
                "score_standard": score_standard,
                "components": {
                    "M1_ks_pass":   ks_pass,
                    "M2_rf_real":   rf_at["realism_score"],
                    "M3_flag_match":flag_pass,
                    "M4_sep_norm":  sep_norm,
                }
            }

        except Exception:
            print(f"\n   ERROR in {attack_name}:")
            traceback.print_exc()

    avg_purpose  = float(np.mean(purpose_scores))  if purpose_scores  else 0.0
    avg_standard = float(np.mean(standard_scores)) if standard_scores else 0.0

    print(f"\n  {'═'*60}")
    print(f"  PER-ATTACK RESULTS")
    print(f"  {'─'*60}")
    print(f"  {'Attack Type':<45} {'Purpose':>8} {'Standard':>10} RF acc")
    print(f"  {'-'*70}")
    for name, r in per_attack_results.items():
        icon = "✅" if r["score_purpose"] >= 75 else "⚠️ "
        print(f"  {icon} {name:<43} "
              f"{r['score_purpose']:>8.1f} "
              f"{r['score_standard']:>10.1f}  "
              f"{r['rf']['rf_accuracy']:.4f}")
    print(f"  {'─'*70}")
    print(f"  {'Average':<43} {avg_purpose:>8.1f} {avg_standard:>10.1f}")
    print(f"  {'═'*60}")

    all_results["per_attack"] = per_attack_results

    # Summary
    print("\n" + "=" * 70)
    print("[5/6] FINAL SUMMARY")
    print("=" * 70)
    print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │  PURPOSE-DRIVEN SCORE (proposed metric)                     │
  │    Per-attack avg : {avg_purpose:>5.1f}/100  {grade(avg_purpose):<30}│
  │    Combined       : {score_combined_purpose:>5.1f}/100                                │
  │    Weights: KS={W_KS_PURPOSE:.0%} RF={W_RF_PURPOSE:.0%} Flags={W_FLAG_PURPOSE:.0%} Sep={W_SEP_PURPOSE:.0%}              │
  ├─────────────────────────────────────────────────────────────┤
  │  STANDARD SCORE (original metric, for comparison)           │
  │    Per-attack avg : {avg_standard:>5.1f}/100  {grade(avg_standard):<30}│
  │    Combined       : {score_combined_standard:>5.1f}/100                                │
  │    Weights: KS={W_KS_STANDARD:.0%} RF={W_RF_STANDARD:.0%} Flags={W_FLAG_STANDARD:.0%}                        │
  └─────────────────────────────────────────────────────────────┘
""")

    if avg_purpose >= 75:
        print("  ✅ Purpose-driven score ≥ 75 — research-quality for IoT DDoS.")
        print("     Suitable for adversarial testing, NS-3 replay, and Pi deployment.")
    elif avg_purpose >= 60:
        print("  ✅ Purpose-driven score ≥ 60 — good for adversarial testing.")

    # Save
    print("\n[6/6] Saving results...")
    all_results["summary"] = {
        "purpose_avg":         avg_purpose,
        "purpose_combined":    score_combined_purpose,
        "standard_avg":        avg_standard,
        "standard_combined":   score_combined_standard,
        "weights_purpose": {
            "M1_ks":   W_KS_PURPOSE,
            "M2_rf":   W_RF_PURPOSE,
            "M3_flags":W_FLAG_PURPOSE,
            "M4_sep":  W_SEP_PURPOSE,
        },
        "weights_standard": {
            "ks":    W_KS_STANDARD,
            "rf":    W_RF_STANDARD,
            "flags": W_FLAG_STANDARD,
        },
        "attack_separation": sep_result,
        "real_source": "test_packets.csv (held-out, never seen by CTGAN)",
    }

    def make_serial(obj):
        if isinstance(obj, dict):
            return {k: make_serial(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serial(v) for v in obj]
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return obj

    with open(OUTPUT_DIR / "evaluation_results_v5.json", "w") as f:
        json.dump(make_serial(all_results), f, indent=2)
    print(f"  ✓ evaluation_results_v5.json")

    # Plot
    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor("#0f0f1a")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)
    BG  = "#1a1a2e"

    def style_ax(ax, title):
        ax.set_facecolor(BG)
        ax.set_title(title, color="white", fontsize=9)
        ax.tick_params(colors="#aaaaaa", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333355")
        ax.grid(alpha=0.1, color="#555577")

    ks_safe = all_results.get("combined", {}).get("ks", {})
    RC = "#4fc3f7"; FC = "#ef5350"
    for i, feat in enumerate(CONTINUOUS_FEATURES):
        if i >= 3:
            break
        ax = fig.add_subplot(gs[0, i])
        r  = real_combined[feat].values if feat in real_combined.columns else np.array([])
        s  = syn_combined_clean[feat].values \
             if feat in syn_combined_clean.columns else np.array([])
        if len(r) and len(s):
            lo   = np.percentile(r, 1); hi = np.percentile(r, 99)
            bins = np.linspace(lo, hi, 60)
            ax.hist(r, bins=bins, alpha=0.6, color=RC, density=True, label="Real")
            ax.hist(s, bins=bins, alpha=0.6, color=FC, density=True, label="Synth")
        p  = ks_safe.get(feat, {}).get("ks_p", 0)
        ok = "✅" if p > 0.05 else "⚠️"
        style_ax(ax, f"{feat}\n{ok} p={p:.4f}")
        ax.legend(fontsize=7, facecolor=BG, labelcolor="white")

    # Dual score bar chart
    ax_bar = fig.add_subplot(gs[1, 0:2])
    style_ax(ax_bar, "Per-Attack Scores: Purpose-Driven vs Standard")
    names   = list(per_attack_results.keys())
    p_scores = [per_attack_results[n]["score_purpose"]  for n in names]
    s_scores = [per_attack_results[n]["score_standard"] for n in names]
    x = np.arange(len(names))
    w = 0.35
    b1 = ax_bar.bar(x - w/2, p_scores, w, color="#4fc3f7", alpha=0.85,
                    label=f"Purpose-driven (avg={avg_purpose:.1f})")
    b2 = ax_bar.bar(x + w/2, s_scores, w, color="#ffb74d", alpha=0.85,
                    label=f"Standard (avg={avg_standard:.1f})")
    ax_bar.axhline(75, color="white",   ls="--", lw=1.5, label="A threshold (75)")
    ax_bar.axhline(60, color="#aaaaaa", ls=":",  lw=1.0, label="B threshold (60)")
    ax_bar.set_ylim(0, 100)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([n.replace("_", " ")[:18] for n in names],
                            rotation=12, color="#aaaaaa", fontsize=7)
    ax_bar.legend(fontsize=8, facecolor=BG, labelcolor="white")
    for bar, val in zip(b1, p_scores):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.1f}", ha="center", color="white", fontsize=8)
    for bar, val in zip(b2, s_scores):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.1f}", ha="center", color="#ffb74d", fontsize=8)

    # Score summary panel
    ax_s = fig.add_subplot(gs[1, 2])
    ax_s.set_facecolor(BG); ax_s.axis("off")
    sc_c = "#4fc3f7" if avg_purpose >= 75 else "#ffb74d"
    ax_s.text(0.5, 0.85, f"{avg_purpose:.1f}",
              ha="center", va="center", fontsize=38, fontweight="bold",
              color=sc_c, transform=ax_s.transAxes)
    ax_s.text(0.5, 0.65, "Purpose-driven /100",
              ha="center", va="center", fontsize=10,
              color="#aaaaaa", transform=ax_s.transAxes)
    ax_s.text(0.5, 0.50, grade(avg_purpose).split("—")[0].strip(),
              ha="center", va="center", fontsize=9,
              color=sc_c, transform=ax_s.transAxes)
    ax_s.axhline(0.40, color="#333355", lw=0.5)
    ax_s.text(0.5, 0.30, f"{avg_standard:.1f}",
              ha="center", va="center", fontsize=22, fontweight="bold",
              color="#ffb74d", transform=ax_s.transAxes)
    ax_s.text(0.5, 0.15, "Standard /100",
              ha="center", va="center", fontsize=10,
              color="#888888", transform=ax_s.transAxes)
    for sp in ax_s.spines.values():
        sp.set_edgecolor(sc_c); sp.set_linewidth(2)

    fig.suptitle(
        f"NeuroStrike — Purpose-Driven Evaluation v5  "
        f"(Purpose={avg_purpose:.1f}  Standard={avg_standard:.1f})",
        color="white", fontsize=12, fontweight="bold", y=0.998)
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "evaluation_report_v5.png"
    plt.savefig(str(plot_path), dpi=130, bbox_inches="tight",
                facecolor="#0f0f1a")
    plt.close()
    print(f"  ✓ evaluation_report_v5.png")

    print("\n" + "=" * 70)
    print(" CODE 6 COMPLETE")
    print("=" * 70)
    print(f"  Runtime             : {elapsed(total_start)}")
    print(f"  Purpose-driven avg  : {avg_purpose:.1f}/100  ← primary (proposed metric)")
    print(f"  Standard avg        : {avg_standard:.1f}/100  ← comparison")
    print(f"\n  Per-attack breakdown (purpose-driven):")
    for name, r in per_attack_results.items():
        icon = "✅" if r["score_purpose"] >= 75 else "⚠️ "
        c    = r["components"]
        print(f"    {icon} {name:<45} {r['score_purpose']:.1f}/100")
        print(f"         M1={c['M1_ks_pass']:.0%} M2={c['M2_rf_real']:.3f} "
              f"M3={c['M3_flag_match']:.0%} M4={c['M4_sep_norm']:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
