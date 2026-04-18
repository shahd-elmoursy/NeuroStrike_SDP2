import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION
BASE_DIR      = Path(r"C:\Users\Lenovo\Downloads\DoS-DDoS-MQTT-IoT_Dataset")
DATA_DIR      = BASE_DIR / "CTGAN_Ready_NeuroStrike_v2"      # test_packets.csv
SYNTHETIC_DIR = BASE_DIR / "FINALCTGAN_Synthetic_Traffic_v2" / "per_attack"
OUTPUT_DIR    = BASE_DIR / "Evaluation_Visual_NeuroStrike"
OUTPUT_DIR.mkdir(exist_ok=True)

# RF score from Code 6 v4 — primary metric, shown in report for context
RF_SCORE_PRIMARY = 62.2

ATTACK_TYPES = [
    "Basic_Connect_Flooding",
    "Connect_Flooding_with_WILL_payload",
    "Delayed_Connect_Flooding",
    "Invalid_Subscription_Flooding",
    "SYN_TCP_Flooding",
]

# Attacks where payload_len was log-transformed — invert when loading test data
PAYLOAD_LOG_ATTACKS = {
    "Connect_Flooding_with_WILL_payload",
    "Invalid_Subscription_Flooding",
}

CONTINUOUS_FEATURES = ['delta_time', 'packet_len', 'payload_len', 'tcp_window_size']
BINARY_FEATURES     = ['flag_syn', 'flag_ack', 'flag_fin', 'flag_rst',
                        'flag_psh', 'flag_urg', 'port_direction']
ALL_FEATURES        = CONTINUOUS_FEATURES + BINARY_FEATURES

KS_SAMPLE    = 10_000
AUTOCORR_LAG = 10

print("=" * 70)
print("NeuroStrike - Evaluation Report (Visual)")
print("=" * 70)
print(f"\nReal data  : {DATA_DIR / 'test_packets.csv'}")
print(f"Synthetic  : {SYNTHETIC_DIR}")
print(f"Output     : {OUTPUT_DIR}")
print(f"RF score (primary, from Code 5): {RF_SCORE_PRIMARY}/100")
print("=" * 70)

# DATA LOADING
def load_real_data() -> pd.DataFrame:
    test_path = DATA_DIR / "test_packets.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"test_packets.csv not found: {test_path}")

    print(f"  Loading test split...")
    test_df = pd.read_csv(test_path)
    print(f"  Total test rows: {len(test_df):,}")

    dfs = []
    for attack in ATTACK_TYPES:
        subset = test_df[test_df["attack_name"] == attack][ALL_FEATURES].copy()
        subset = subset.dropna(subset=ALL_FEATURES)
        if len(subset) == 0:
            print(f"    ⚠️  No test rows for {attack}")
            continue

        # Invert log transforms (same as Code 6)
        subset["delta_time"] = (
            np.expm1(subset["delta_time"].values.astype(float)) / 1000.0
        )
        subset["delta_time"] = subset["delta_time"].clip(0, 10.0)

        if attack in PAYLOAD_LOG_ATTACKS:
            subset["payload_len"] = np.expm1(
                subset["payload_len"].values.astype(float))
            subset["payload_len"] = subset["payload_len"].clip(0, 1460)

        subset["attack_type"] = attack
        dfs.append(subset)
        print(f"    ✓ {attack}: {len(subset):,} test packets (log-inverted)")

    return pd.concat(dfs, ignore_index=True)


def load_synthetic_data() -> pd.DataFrame:
    dfs = []
    for attack in ATTACK_TYPES:
        f = SYNTHETIC_DIR / f"{attack}_synthetic.csv"
        if not f.exists():
            print(f"      Not found: {f.name}")
            continue
        df = pd.read_csv(f, usecols=ALL_FEATURES)
        df = df.dropna(subset=ALL_FEATURES)
        df["attack_type"] = attack
        dfs.append(df)
        print(f"    ✓ {attack}: {len(df):,} synthetic rows")
    return pd.concat(dfs, ignore_index=True)


print("\n[1/5] Loading data...")
print("-" * 70)
print("  Real data (test split, log-inverted):")
real_df = load_real_data()
print(f"  ✓ Real total: {len(real_df):,} packets\n")
print("  Synthetic data (Code 4):")
syn_df = load_synthetic_data()
print(f"  ✓ Synthetic total: {len(syn_df):,} packets")

real_df = real_df.dropna(subset=ALL_FEATURES)
syn_df  = syn_df.dropna(subset=ALL_FEATURES)
print(f"\n  After cleaning — Real: {len(real_df):,}  Synthetic: {len(syn_df):,}")

# STATISTICAL ANALYSIS
print("\n[2/5] Computing statistics...")
print("-" * 70)


def compute_autocorr(data, max_lag=10):
    data = np.asarray(data, dtype=float)
    mean = np.mean(data)
    var  = np.var(data)
    if var == 0:
        return [0.0] * max_lag
    result = []
    for lag in range(1, max_lag + 1):
        if len(data) > lag:
            result.append(
                float(np.mean((data[:-lag] - mean) * (data[lag:] - mean)) / var))
    return result


def feature_stats(arr):
    arr = np.asarray(arr, dtype=float)
    return {
        'mean':   float(np.mean(arr)),
        'std':    float(np.std(arr)),
        'median': float(np.median(arr)),
        'min':    float(np.min(arr)),
        'max':    float(np.max(arr)),
        'p25':    float(np.percentile(arr, 25)),
        'p75':    float(np.percentile(arr, 75)),
    }


results = {}
for feat in ALL_FEATURES:
    r_vals = real_df[feat].values
    s_vals = syn_df[feat].values
    r_st   = feature_stats(r_vals)
    s_st   = feature_stats(s_vals)

    n  = min(KS_SAMPLE, len(r_vals), len(s_vals))
    ks = stats.ks_2samp(
        np.random.choice(r_vals, n, replace=False),
        np.random.choice(s_vals, n, replace=False)
    )

    mean_range = max(abs(r_st['max'] - r_st['min']), 1e-9)
    mean_sim   = float(max(0, 1 - abs(r_st['mean'] - s_st['mean']) / mean_range) * 100)
    std_sim    = float(max(0, 1 - abs(r_st['std']  - s_st['std'])  /
                            max(r_st['std'], 1e-9)) * 100)

    results[feat] = {
        'real':        r_st,
        'synthetic':   s_st,
        'ks_stat':     float(ks.statistic),
        'ks_pvalue':   float(ks.pvalue),
        'mean_sim':    mean_sim,
        'std_sim':     std_sim,
        'feature_sim': float((mean_sim + std_sim) / 2),
    }

    label = '✅' if ks.pvalue > 0.05 else '⚠️ '
    print(f"  {label} {feat:<22} KS={ks.statistic:.4f}  p={ks.pvalue:.4f}  "
          f"sim={results[feat]['feature_sim']:.1f}%")

# Autocorrelation on delta_time
r_dt       = real_df['delta_time'].values[:KS_SAMPLE]
s_dt       = syn_df['delta_time'].values[:KS_SAMPLE]
r_autocorr = compute_autocorr(r_dt, AUTOCORR_LAG)
s_autocorr = compute_autocorr(s_dt, AUTOCORR_LAG)
autocorr_diffs = [abs(r - s) for r, s in zip(r_autocorr, s_autocorr)]
autocorr_sim   = float(max(0, 1 - np.mean(autocorr_diffs)) * 100)

# Flag rates
flag_rate_real = {f: float(real_df[f].mean()) for f in BINARY_FEATURES}
flag_rate_syn  = {f: float(syn_df[f].mean())  for f in BINARY_FEATURES}

print(f"\n  Flag rates:")
for f in BINARY_FEATURES:
    r = flag_rate_real[f]
    s = flag_rate_syn[f]
    ok = '✅' if abs(r - s) < 0.05 else '⚠️ '
    print(f"  {ok} {f:<18} real={r:.3f}  syn={s:.3f}  diff={abs(r-s):.3f}")

print(f"\n  Temporal (autocorr) similarity: {autocorr_sim:.1f}%")

continuous_sims = [results[f]['feature_sim'] for f in CONTINUOUS_FEATURES]
binary_sims     = [results[f]['feature_sim'] for f in BINARY_FEATURES]
visual_score    = float(np.mean(continuous_sims + binary_sims + [autocorr_sim]))

print(f"\n  ══════════════════════════════════════════════")
print(f"  VISUAL SIMILARITY SCORE : {visual_score:.1f}%")
print(f"  RF DISCRIMINATOR SCORE  : {RF_SCORE_PRIMARY}/100  ← primary metric")
print(f"  ══════════════════════════════════════════════")
print(f"\n  NOTE: Visual score uses mean+std similarity (lenient).")
print(f"        RF score uses a classifier — much stricter.")
print(f"        Report BOTH in your paper with this explanation.")

# VISUALIZATION
print("\n[3/5] Generating visualization...")
print("-" * 70)

REAL_COLOR = '#4fc3f7'
SYN_COLOR  = '#ef5350'
BG_DARK    = '#0f0f1a'
BG_PANEL   = '#1a1a2e'
BG_ALT     = '#16213e'


def style_ax(ax, title):
    ax.set_facecolor(BG_PANEL)
    ax.set_title(title, color='white', fontsize=9, fontweight='bold', pad=6)
    ax.tick_params(colors='#aaaaaa', labelsize=7)
    ax.xaxis.label.set_color('#aaaaaa')
    ax.yaxis.label.set_color('#aaaaaa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')
    ax.grid(alpha=0.15, color='#555577')


fig = plt.figure(figsize=(24, 30))
fig.patch.set_facecolor(BG_DARK)
gs  = gridspec.GridSpec(5, 4, figure=fig, hspace=0.5, wspace=0.35)

# Row 0: Histograms for 4 continuous features
for i, feat in enumerate(CONTINUOUS_FEATURES):
    ax   = fig.add_subplot(gs[0, i])
    r    = real_df[feat].values
    s    = syn_df[feat].values
    lo   = min(np.percentile(r, 1),  np.percentile(s, 1))
    hi   = max(np.percentile(r, 99), np.percentile(s, 99))
    bins = np.linspace(lo, hi, 80)
    ax.hist(r, bins=bins, alpha=0.55, color=REAL_COLOR, density=True, label='Real')
    ax.hist(s, bins=bins, alpha=0.55, color=SYN_COLOR,  density=True, label='Synthetic')
    style_ax(ax, f'{feat}\nKS={results[feat]["ks_stat"]:.3f}  '
                 f'sim={results[feat]["feature_sim"]:.1f}%')
    ax.set_ylabel('Density', fontsize=7)
    ax.legend(fontsize=7, facecolor=BG_PANEL, labelcolor='white')

# Row 1: CDFs for 4 continuous features
for i, feat in enumerate(CONTINUOUS_FEATURES):
    ax  = fig.add_subplot(gs[1, i])
    r_s = np.sort(real_df[feat].values)
    s_s = np.sort(syn_df[feat].values)
    ax.plot(r_s, np.linspace(0, 1, len(r_s)),
            color=REAL_COLOR, lw=1.8, label='Real')
    ax.plot(s_s, np.linspace(0, 1, len(s_s)),
            color=SYN_COLOR, lw=1.8, ls='--', label='Synthetic')
    style_ax(ax, f'CDF: {feat}')
    ax.set_ylabel('Cumulative prob.', fontsize=7)
    ax.legend(fontsize=7, facecolor=BG_PANEL, labelcolor='white')

# Row 2: Flag rates | Autocorrelation | Similarity bars | Dual score
ax_flags = fig.add_subplot(gs[2, 0:2])
x = np.arange(len(BINARY_FEATURES))
w = 0.35
b_r = ax_flags.bar(x - w/2, [flag_rate_real[f] for f in BINARY_FEATURES],
                    w, color=REAL_COLOR, alpha=0.8, label='Real')
b_s = ax_flags.bar(x + w/2, [flag_rate_syn[f]  for f in BINARY_FEATURES],
                    w, color=SYN_COLOR,  alpha=0.8, label='Synthetic')
ax_flags.set_xticks(x)
ax_flags.set_xticklabels(
    [f.replace('flag_', '').replace('port_direction', 'port_dir')
     for f in BINARY_FEATURES],
    color='#aaaaaa', fontsize=8)
style_ax(ax_flags, 'Binary Feature Rates: Real vs Synthetic')
ax_flags.set_ylabel('Rate')
ax_flags.legend(fontsize=8, facecolor=BG_PANEL, labelcolor='white')

ax_auto = fig.add_subplot(gs[2, 2])
lags = list(range(1, len(r_autocorr) + 1))
ax_auto.plot(lags, r_autocorr, 'o-', color=REAL_COLOR, lw=2, ms=5, label='Real')
ax_auto.plot(lags, s_autocorr, 's--', color=SYN_COLOR,  lw=2, ms=5, label='Synthetic')
ax_auto.axhline(0, color='#555577', lw=0.8)
style_ax(ax_auto, f'Autocorrelation: delta_time\n'
                   f'temporal sim={autocorr_sim:.1f}%')
ax_auto.set_xlabel('Lag', fontsize=7)
ax_auto.set_ylabel('Autocorrelation', fontsize=7)
ax_auto.legend(fontsize=7, facecolor=BG_PANEL, labelcolor='white')

ax_score = fig.add_subplot(gs[2, 3])
sc_labels = CONTINUOUS_FEATURES + ['temporal']
sc_vals   = continuous_sims + [autocorr_sim]
sc_colors = ['#4fc3f7', '#81c784', '#ffb74d', '#ce93d8', '#f48fb1']
bars_sc   = ax_score.barh(sc_labels, sc_vals, color=sc_colors, alpha=0.85)
ax_score.set_xlim(0, 100)
ax_score.axvline(visual_score, color='white', ls='--', lw=1.5,
                 label=f'Visual {visual_score:.1f}%')
style_ax(ax_score, 'Feature Similarity (Visual metric)')
ax_score.set_xlabel('Similarity %', fontsize=7)
ax_score.legend(fontsize=7, facecolor=BG_PANEL, labelcolor='white')
for bar, val in zip(bars_sc, sc_vals):
    ax_score.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                  f'{val:.1f}%', va='center', color='white', fontsize=7)

# Row 3: Box plots
for i, feat in enumerate(CONTINUOUS_FEATURES):
    ax = fig.add_subplot(gs[3, i])
    lo = np.percentile(real_df[feat].values, 1)
    hi = np.percentile(real_df[feat].values, 99)
    r  = real_df[feat].clip(lo, hi).values
    s  = syn_df[feat].clip(lo, hi).values
    for pos, arr, col in [(1, r, REAL_COLOR), (2, s, SYN_COLOR)]:
        ax.boxplot(arr, positions=[pos], widths=0.5, patch_artist=True,
                   boxprops=dict(facecolor=col, alpha=0.6),
                   medianprops=dict(color='white', linewidth=2),
                   whiskerprops=dict(color='#aaaaaa'),
                   capprops=dict(color='#aaaaaa'),
                   flierprops=dict(marker='.', color=col, alpha=0.2, markersize=2))
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Real', 'Synthetic'], color='#aaaaaa', fontsize=8)
    style_ax(ax, f'Box plot: {feat}')
    ax.set_ylabel('Value', fontsize=7)

# Row 4: Summary table + dual score verdict
ax_table = fig.add_subplot(gs[4, 0:3])
ax_table.axis('off')
table_rows = []
for feat in ALL_FEATURES:
    r = results[feat]
    table_rows.append([
        feat,
        f"{r['real']['mean']:.4f}",
        f"{r['synthetic']['mean']:.4f}",
        f"{r['real']['std']:.4f}",
        f"{r['synthetic']['std']:.4f}",
        f"{r['ks_stat']:.4f}",
        f"{r['ks_pvalue']:.4f}",
        f"{r['feature_sim']:.1f}%",
    ])
col_labels = ['Feature', 'Real μ', 'Syn μ', 'Real σ',
              'Syn σ', 'KS stat', 'KS p', 'Sim %']
table = ax_table.table(cellText=table_rows, colLabels=col_labels,
                       loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.5)
for (row, col), cell in table.get_celld().items():
    cell.set_facecolor(BG_PANEL if row % 2 == 0 else BG_ALT)
    cell.set_text_props(color='white')
    cell.set_edgecolor('#333355')
    if row == 0:
        cell.set_facecolor('#0d3b66')
        cell.set_text_props(color='white', fontweight='bold')
style_ax(ax_table, 'Full Feature Comparison (11 features, Code 4 output)')

# Dual score verdict panel
ax_v = fig.add_subplot(gs[4, 3])
ax_v.axis('off')
ax_v.set_facecolor(BG_DARK)

# RF score (primary)
ax_v.text(0.5, 0.90, "RF score (primary)",
          ha='center', va='center', fontsize=8,
          color='#aaaaaa', transform=ax_v.transAxes)
rf_color = '#4fc3f7' if RF_SCORE_PRIMARY >= 60 else '#ffb74d'
ax_v.text(0.5, 0.74, f"{RF_SCORE_PRIMARY}/100",
          ha='center', va='center', fontsize=28, fontweight='bold',
          color=rf_color, transform=ax_v.transAxes)
ax_v.text(0.5, 0.60, "B — good for adversarial testing",
          ha='center', va='center', fontsize=8,
          color=rf_color, transform=ax_v.transAxes)

# Divider
ax_v.axhline(0.52, color='#333355', lw=0.5)

# Visual score (supplementary)
ax_v.text(0.5, 0.44, "Visual score (supplementary)",
          ha='center', va='center', fontsize=8,
          color='#aaaaaa', transform=ax_v.transAxes)
vis_color = '#66bb6a' if visual_score >= 75 else '#ffb74d'
ax_v.text(0.5, 0.28, f"{visual_score:.1f}%",
          ha='center', va='center', fontsize=28, fontweight='bold',
          color=vis_color, transform=ax_v.transAxes)
ax_v.text(0.5, 0.12, "mean + std similarity",
          ha='center', va='center', fontsize=8,
          color='#888888', transform=ax_v.transAxes)

for spine in ax_v.spines.values():
    spine.set_edgecolor(rf_color)
    spine.set_linewidth(2)
style_ax(ax_v, 'Evaluation Scores')

fig.suptitle(
    f'NeuroStrike CTGAN — Evaluation Report  '
    f'(RF={RF_SCORE_PRIMARY}/100  Visual={visual_score:.1f}%)',
    fontsize=13, fontweight='bold', color='white', y=0.998)

png_path = OUTPUT_DIR / "NeuroStrike_Evaluation_Report_v2.png"
plt.savefig(str(png_path), dpi=150, bbox_inches='tight', facecolor=BG_DARK)
plt.close()
print(f"  ✓ PNG saved: {png_path.name}")

# SAVE JSON + CSV
print("\n[4/5] Saving JSON results...")

json_data = {
    'scores': {
        'rf_score_primary':   RF_SCORE_PRIMARY,
        'rf_grade':           'B — good for adversarial testing',
        'visual_score':       visual_score,
        'note': (
            'RF score is the primary metric (Code 6, RF discriminator). '
            'Visual score uses mean+std similarity — more lenient, supplementary only.'
        ),
    },
    'summary': {
        'real_packets':       int(len(real_df)),
        'synthetic_packets':  int(len(syn_df)),
        'real_source':        'test_packets.csv (held-out, never seen by CTGAN)',
        'synthetic_source':   'FINALCTGAN_Synthetic_Traffic_v2/per_attack (Code 4)',
        'continuous_sim':     float(np.mean(continuous_sims)),
        'binary_sim':         float(np.mean(binary_sims)),
        'temporal_sim':       autocorr_sim,
    },
    'per_feature': {
        feat: {
            'real_mean':      results[feat]['real']['mean'],
            'real_std':       results[feat]['real']['std'],
            'syn_mean':       results[feat]['synthetic']['mean'],
            'syn_std':        results[feat]['synthetic']['std'],
            'ks_statistic':   results[feat]['ks_stat'],
            'ks_pvalue':      results[feat]['ks_pvalue'],
            'similarity_pct': results[feat]['feature_sim'],
        }
        for feat in ALL_FEATURES
    },
    'flag_rates': {
        f: {
            'real':      flag_rate_real[f],
            'synthetic': flag_rate_syn[f],
            'diff':      abs(flag_rate_real[f] - flag_rate_syn[f]),
            'passed':    abs(flag_rate_real[f] - flag_rate_syn[f]) < 0.05,
        }
        for f in BINARY_FEATURES
    },
    'autocorrelation': {
        'real': r_autocorr, 'synthetic': s_autocorr,
        'similarity': autocorr_sim,
    },
}

json_path = OUTPUT_DIR / "NeuroStrike_Evaluation_Results_v2.json"
with open(json_path, 'w') as f:
    json.dump(json_data, f, indent=2)
print(f"  ✓ JSON: {json_path.name}")

print("\n[5/5] Saving CSV summary...")
csv_rows = []
for feat in ALL_FEATURES:
    r = results[feat]
    csv_rows.append({
        'feature':        feat,
        'type':           'continuous' if feat in CONTINUOUS_FEATURES else 'binary',
        'real_mean':      r['real']['mean'],
        'real_std':       r['real']['std'],
        'syn_mean':       r['synthetic']['mean'],
        'syn_std':        r['synthetic']['std'],
        'ks_statistic':   r['ks_stat'],
        'ks_pvalue':      r['ks_pvalue'],
        'ks_pass':        r['ks_pvalue'] > 0.05,
        'similarity_pct': r['feature_sim'],
    })
csv_path = OUTPUT_DIR / "NeuroStrike_Evaluation_Summary_v2.csv"
pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
print(f"  ✓ CSV: {csv_path.name}")

# FINAL SUMMARY
print("\n" + "=" * 70)
print(" EVALUATION COMPLETE")
print("=" * 70)
print(f"\n  Real    : {len(real_df):,} packets (test split)")
print(f"  Synthetic: {len(syn_df):,} packets (Code 4)")
print(f"\n  {'Feature':<22} {'Real μ':>10} {'Syn μ':>10} {'KS p':>8} {'Sim %':>8}")
print(f"  {'-'*62}")
for feat in ALL_FEATURES:
    r    = results[feat]
    flag = '✅' if r['ks_pvalue'] > 0.05 else '⚠️ '
    print(f"  {flag} {feat:<20} {r['real']['mean']:>10.4f} "
          f"{r['synthetic']['mean']:>10.4f} "
          f"{r['ks_pvalue']:>8.4f} {r['feature_sim']:>7.1f}%")
print(f"  {'─'*62}")
print(f"  Continuous sim : {np.mean(continuous_sims):.1f}%")
print(f"  Binary sim     : {np.mean(binary_sims):.1f}%")
print(f"  Temporal sim   : {autocorr_sim:.1f}%")
print(f"  {'─'*62}")
print(f"  Visual score   : {visual_score:.1f}%  (mean+std similarity, supplementary)")
print(f"  RF score       : {RF_SCORE_PRIMARY}/100  (discriminator-based, PRIMARY)")
print(f"  {'─'*62}")
print(f"\n  Outputs: {OUTPUT_DIR}")
print(f"    {png_path.name}")
print(f"    {json_path.name}")
print(f"    {csv_path.name}")
print("=" * 70)
