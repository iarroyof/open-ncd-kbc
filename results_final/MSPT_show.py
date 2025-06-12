#!/usr/bin/env python3
"""
Compare two cosine-similarity samples, plot histograms + KDE, run a t-test.

Gap-percentage options
----------------------
    symmetric  : 2·|μ₁-μ₂| / (μ₁+μ₂)  ×100   ← default (scale-free, symmetric)
    absolute   : |μ₁-μ₂|              ×100   ← simple “percentage points”

Example
-------
python script.py --file1 sample1.txt --file2 sample2.txt \
                 --gap-mode absolute \
                 --name1 "Method A" --name2 "Method B" --bins 25
"""
import numpy as np
import argparse, os, sys
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import gaussian_kde


# ─────────────────────────── helpers ────────────────────────────
def format_p(p: float) -> str:
    """Format p-value in fixed-point if ≥1e-4 else scientific notation."""
    return f"{p:.4f}" if p >= 1e-4 else f"{p:.2e}"


def load_data(path: str) -> np.ndarray:
    """Load numeric values from a text file, filtering out invalid entries."""
    vals, lines = [], 0
    try:
        with open(path, encoding="utf-8") as f:
            for ln in f:
                lines += 1
                ln = ln.strip()
                if not ln or ln.startswith(("#", "//")):
                    continue
                for part in ln.split():
                    try:
                        v = float(part)
                        if np.isnan(v) or np.isinf(v):
                            continue
                        if -0.1 <= v <= 1.1:
                            vals.append(max(0.0, min(1.0, v)))
                    except (ValueError, OverflowError):
                        continue
        if not vals:
            print(f"Warning: no numeric values in {path} ({lines} lines read)")
            return np.array([])
        arr = np.array(vals)
        print(f"Extracted {len(arr)} values from {lines} lines "
              f"([{arr.min():.4f}, {arr.max():.4f}])")
        return arr
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return np.array([])


def ttest(a: np.ndarray, b: np.ndarray):
    """Two-sample independent t-test."""
    return stats.ttest_ind(a, b)


def calc_gap(mu1: float, mu2: float, mode: str = "symmetric") -> float:
    """Compute percentage gap between two means."""
    diff = abs(mu1 - mu2)
    if mode == "absolute":
        return diff * 100.0
    # symmetric default: 2·diff/(mu1+mu2)×100
    return diff / ((mu1 + mu2) / 2) * 100.0


# ─────────────────────────── plotting ───────────────────────────
def build_plot(s1: np.ndarray, s2: np.ndarray,
               name1: str, name2: str,
               bins: int, gap_mode: str) -> go.Figure:
    mu1, mu2 = s1.mean(), s2.mean()
    sd1, sd2 = s1.std(), s2.std()
    t_stat, p_val = ttest(s1, s2)
    gap = calc_gap(mu1, mu2, gap_mode)

    fig = go.Figure()

    # colors
    hist_colors = ["rgba(31,119,180,0.6)", "rgba(255,127,14,0.6)"]
    line_colors = ["rgb(31,119,180)",    "rgb(255,127,14)"]

    # histograms
    fig.add_histogram(x=s1, nbinsx=bins, name=name1,
                      marker_color=hist_colors[0],
                      histnorm="probability density", opacity=0.6)
    fig.add_histogram(x=s2, nbinsx=bins, name=name2,
                      marker_color=hist_colors[1],
                      histnorm="probability density", opacity=0.6)

    # KDE curves
    x_vals = np.linspace(0, 1, 200)
    for data, color, label in [
        (s1, line_colors[0], f"{name1} KDE"),
        (s2, line_colors[1], f"{name2} KDE")
    ]:
        kde = gaussian_kde(data)
        fig.add_scatter(x=x_vals, y=kde(x_vals), mode="lines",
                        line=dict(color=color, width=2), name=label)

    # ─── Mean lines & adaptive label rows ──────────────────────────────
    # maximum density for positioning
    y_max = max(gaussian_kde(s1)(x_vals).max(),
                gaussian_kde(s2)(x_vals).max())

    # if means are within this threshold, split labels into two rows
    close_thresh = 0.05
    close = abs(mu1 - mu2) < close_thresh
    row_fracs = [0.95, 0.85] if close else [0.90, 0.90]
    row_y = [y_max * f for f in row_fracs]

    # μ₁ line + label
    fig.add_vline(x=mu1,
                  line=dict(color=line_colors[0], width=2, dash="dot"))
    fig.add_annotation(
        x=mu1, y=row_y[0],
        text=f"μ₁ = {mu1:.3f}",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=line_colors[0], borderwidth=1,
        font=dict(size=10), yanchor="bottom"
    )

    # μ₂ line + label
    fig.add_vline(x=mu2,
                  line=dict(color=line_colors[1], width=2, dash="dot"))
    fig.add_annotation(
        x=mu2, y=row_y[1],
        text=f"μ₂ = {mu2:.3f}",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=line_colors[1], borderwidth=1,
        font=dict(size=10), yanchor="bottom"
    )

    # gap & p-value annotation
    fig.add_annotation(
        x=(mu1 + mu2) / 2, y=y_max * 0.5,
        text=f"Gap: {gap:.2f}%<br>p-value: {format_p(p_val)}",
        showarrow=True, arrowhead=2,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="gray", borderwidth=1,
        font=dict(size=10)
    )

    # stats summary box
    stats_txt = (
        "<b>Statistical summary</b><br>"
        f"{name1}: μ={mu1:.3f}, σ={sd1:.3f}, n={len(s1)}<br>"
        f"{name2}: μ={mu2:.3f}, σ={sd2:.3f}, n={len(s2)}<br>"
        f"t-stat = {t_stat:.3f}, p = {format_p(p_val)}"
    )
    fig.add_annotation(
        x=0.98, y=0.35, xref="paper", yref="paper",
        text=stats_txt, showarrow=False, align="left",
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="gray", borderwidth=1,
        font=dict(size=9), xanchor="right", yanchor="bottom"
    )

    # layout
    fig.update_layout(
        title=dict(text=f"{name1} vs {name2}", x=0.5),
        xaxis=dict(title="Cosine similarity", range=[0, 1], tickformat=".2f"),
        yaxis=dict(title="Probability density"),
        barmode="overlay",
        template="plotly_white",
        width=1000, height=700
    )

    return fig


# ─────────────────────────── main ─────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate comparative histogram + KDE and run t-test"
    )
    parser.add_argument("--file1",    help="Path to first sample file")
    parser.add_argument("--file2",    help="Path to second sample file")
    parser.add_argument("--name1",    default="Sample 1",
                        help="Display name for first sample")
    parser.add_argument("--name2",    default="Sample 2",
                        help="Display name for second sample")
    parser.add_argument("--bins",     type=int, default=30,
                        help="Number of histogram bins")
    parser.add_argument("--output",   default="comparison_plot.pdf",
                        help="Output PDF filename")
    parser.add_argument("--gap-mode", choices=["symmetric", "absolute"],
                        default="symmetric",
                        help="How to compute percentage gap")
    args = parser.parse_args()

    f1 = args.file1 or os.getenv("ANALYSIS_FILE1")
    f2 = args.file2 or os.getenv("ANALYSIS_FILE2")
    if not (f1 and f2):
        parser.error("Provide --file1/--file2 or set ANALYSIS_FILE1/2 env vars.")

    s1 = load_data(f1)
    s2 = load_data(f2)
    if len(s1) == 0 or len(s2) == 0:
        sys.exit("Error: one or both files contain no valid measurements.")

    print(f"\n{args.name1}: μ={s1.mean():.3f}, σ={s1.std():.3f}")
    print(f"{args.name2}: μ={s2.mean():.3f}, σ={s2.std():.3f}")

    fig = build_plot(s1, s2, args.name1, args.name2, args.bins, args.gap_mode)
    print(f"Saving figure to {args.output} …")
    fig.write_image(args.output, format="pdf", engine="kaleido")
    print("Done!")

    t_stat, p_val = ttest(s1, s2)
    gap = calc_gap(s1.mean(), s2.mean(), args.gap_mode)
    print(f"\nt-statistic = {t_stat:.4f}")
    print(f"p-value     = {format_p(p_val)}")
    print(f"gap         = {gap:.2f}% ({args.gap_mode} mode)")


if __name__ == "__main__":
    main()

""" 
# Usage examples:
# For the best balanced results of the best regularized model 

cd open-ncd-kbc/results
python MSPT_show.py --name1 Val_preds_to_random_baseline_STS \
    --name2 Val_preds_to_true_object_STS \
    --file1 open-ncd-kbc/ushzkb2o/1dtkesgi/val_object_pairs_predictions_random.tsv.output_fstx_300d_indexed_sum_tfidf_none_idf \
    --file2 open-ncd-kbc/ushzkb2o/1dtkesgi/val_object_pairs_predictions.tsv.output_fstx_300d_indexed_sum_tfidf_none_idf \
    --output open-ncd-kbc/ushzkb2o/1dtkesgi/histogram_val.pdf --gap-mode absolute
python MSPT_show.py --name1 Test_preds_to_random_baseline_STS \
    --name2 Test_preds_to_true_object_STS --file1 open-ncd-kbc/ushzkb2o/1dtkesgi/test_object_pairs_predictions_random.tsv.output_fstx_300d_indexed_sum_tfidf_none_idf \
    --file2 open-ncd-kbc/ushzkb2o/1dtkesgi/test_object_pairs_predictions.tsv.output_fstx_300d_indexed_sum_tfidf_none_idf \
    --output open-ncd-kbc/ushzkb2o/1dtkesgi/histogram_test.pdf --gap-mode absolute

# For the globaly best gru model
cd AIgroKB/results_final
python MSPT_show.py --name1 Test_preds_to_random_baseline_STS \
    --name2 Test_preds_to_true_object_STS \
    --file1 CSRncdKBC-attentionGRU_epochs-100_seqlen-10_maxfeat-15000_batch-128_embdim-256_steps-2048\object_pairs_random.tsv.output_fstx_300d_indexed_sum_tfidf_none_idf \
    --file2 CSRncdKBC-attentionGRU_epochs-100_seqlen-10_maxfeat-15000_batch-128_embdim-256_steps-2048\object_pairs.tsv.output_fstx_300d_indexed_sum_tfidf_none_idf \
    --output CSRncdKBC-attentionGRU_epochs-100_seqlen-10_maxfeat-15000_batch-128_embdim-256_steps-2048\histogram_test.pdf \
    --gap-mode absolute

python MSPT_show.py --name1 Val_preds_to_random_baseline_STS \
    --name2 Val_preds_to_true_object_STS \
    --file1 CSRncdKBC-attentionGRU_epochs-100_seqlen-10_maxfeat-15000_batch-128_embdim-256_steps-2048\object_pairs_val_random.tsv.output_fstx_300d_indexed_sum_tfidf_none_idf \
    --file2 CSRncdKBC-attentionGRU_epochs-100_seqlen-10_maxfeat-15000_batch-128_embdim-256_steps-2048\object_pairs_val.tsv.output_fstx_300d_indexed_sum_tfidf_none_idf \
    --output CSRncdKBC-attentionGRU_epochs-100_seqlen-10_maxfeat-15000_batch-128_embdim-256_steps-2048\histogram_val.pdf \
    --gap-mode absolute
    
# For the best transformer
cd AIgroKB/results_final
python MSPT_show.py --name1 Test_preds_to_random_baseline_STS \
    --name2 Test_preds_to_true_object_STS \
    --file1 ncd-conceptnet-transformer_epochs-40_stackSize-1_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8\object_pairs_random.tsv.output_fstx_300d_indexed_sum_tfidf_none_idf \
    --file2 ncd-conceptnet-transformer_epochs-40_stackSize-1_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8\object_pairs.tsv.output_fstx_300d_indexed_sum_tfidf_none_idf
    --output ncd-conceptnet-transformer_epochs-40_stackSize-1_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8\histogram_test.pdf \
    -gap-mode absolute
    
python MSPT_show.py --name1 Val_preds_to_random_baseline_STS \
    --name2 Val_preds_to_true_object_STS \
    --file1 ncd-conceptnet-transformer_epochs-40_stackSize-1_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8\object_pairs_val_random.tsv.output_fstx_300d_indexed_sum_tfidf_none_idf \
    --file2 ncd-conceptnet-transformer_epochs-40_stackSize-1_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8\object_pairs_val.tsv.output_fstx_300d_indexed_sum_tfidf_none_idf \
    --output ncd-conceptnet-transformer_epochs-40_stackSize-1_seqlen-30_maxfeat-15000_batch-64_keydim-64_modeldim-512_latent-2048_heads-8\histogram_val.pdf \
    -gap-mode absolute
    
"""