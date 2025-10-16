import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# === Configuration ===
TOP_N = 6          # Number of unique papers to display per theme (top one is locked)
COLUMNS = 2        # Subplots per row
FIG_W = 19
ROW_H = 4.0

# --- Load data ---
df = pd.read_csv("classified_papers_semantic_weighted.csv")

# Validate & normalize columns
if 'Paper Title' not in df.columns:
    raise ValueError("CSV must contain a 'Paper Title' column.")
df['Paper Title'] = df['Paper Title'].astype(str)

# Extract theme score columns (ending with " (score)")
score_cols = [c for c in df.columns if c.endswith(" (score)")]
themes = [c[:-8] for c in score_cols]  # strip " (score)"

# Build score matrix: rows = themes, cols = papers
NEG_INF = -1e12
scores = np.vstack([df[f"{t} (score)"].astype(float).to_numpy() for t in themes]).astype(float)
scores = np.where(np.isnan(scores), NEG_INF, scores)

n_themes, n_papers = scores.shape
paper_titles = df['Paper Title'].tolist()

# === Step 1: Lock the unique TOP-1 paper per theme (never changes with TOP_N) ===
locked_theme_to_paper = {}

def optimal_unique_top1(scores_matrix):
    """Return dict: theme_index -> paper_index using Hungarian to maximize total score (fallback: greedy)."""
    mapping = {}
    try:
        from scipy.optimize import linear_sum_assignment
        n_t, n_p = scores_matrix.shape
        # Pad with dummy papers if fewer papers than themes
        if n_p < n_t:
            pad = np.full((n_t, n_t - n_p), NEG_INF, dtype=float)
            sp = np.hstack([scores_matrix, pad])
            titles_padded = paper_titles + [f"__DUMMY_{i}__" for i in range(n_t - n_p)]
        else:
            sp = scores_matrix
            titles_padded = paper_titles

        # Convert to costs for maximization
        shift = np.max(sp)
        if not np.isfinite(shift):
            shift = 0.0
        cost = shift - sp
        row_ind, col_ind = linear_sum_assignment(cost)

        for r, c in zip(row_ind, col_ind):
            if c >= len(titles_padded):
                continue
            if titles_padded[c].startswith("__DUMMY_"):
                continue
            if sp[r, c] <= NEG_INF/10:
                continue
            mapping[r] = c
        return mapping
    except Exception:
        # Deterministic greedy fallback
        triples = []
        for ti in range(scores_matrix.shape[0]):
            for pi in range(scores_matrix.shape[1]):
                s = float(scores_matrix[ti, pi])
                triples.append((s, ti, pi))
        triples.sort(key=lambda x: (-x[0], x[1], x[2]))
        used_t, used_p = set(), set()
        for s, ti, pi in triples:
            if s <= NEG_INF/10: 
                continue
            if ti in used_t or pi in used_p:
                continue
            mapping[ti] = pi
            used_t.add(ti)
            used_p.add(pi)
            if len(used_t) == scores_matrix.shape[0]:
                break
        return mapping

# Lock the single best unique paper for each theme
locked_theme_to_paper = optimal_unique_top1(scores)

# === Step 2: Fill remaining (TOP_N-1) slots with uniqueness preserved, without altering locked picks ===
assigned = {theme: [] for theme in themes}  # theme -> list of (paper_idx, score)
used_papers = set()

# Seed with locked picks
for ti, theme in enumerate(themes):
    pi = locked_theme_to_paper.get(ti, None)
    if pi is not None:
        assigned[theme].append((pi, float(scores[ti, pi])))
        used_papers.add(pi)

# If TOP_N == 1 we're done; otherwise, globally greedy-fill the rest
if TOP_N > 1:
    # Build all remaining (score, theme_idx, paper_idx) excluding already-used papers
    triples = []
    for ti, theme in enumerate(themes):
        for pi in range(n_papers):
            if pi in used_papers:
                continue
            s = float(scores[ti, pi])
            if s <= NEG_INF/10:
                continue
            triples.append((s, ti, pi))
    # Sort by score desc with deterministic tie-breakers
    triples.sort(key=lambda x: (-x[0], x[1], x[2]))

    needed = {theme: max(0, TOP_N - len(assigned[theme])) for theme in themes}
    remaining_needed = sum(needed.values())

    for s, ti, pi in triples:
        if remaining_needed == 0:
            break
        theme = themes[ti]
        if needed[theme] <= 0:
            continue
        if pi in used_papers:
            continue
        assigned[theme].append((pi, s))
        used_papers.add(pi)
        needed[theme] -= 1
        remaining_needed -= 1

# --- Build tidy per-theme tables and compute global x-max ---
all_selected_scores = []
per_theme_rows = {}
for theme_i, theme in enumerate(themes):
    # Sort selections for this theme (high to low) and cap to TOP_N (defensive)
    sel = sorted(assigned[theme], key=lambda x: -x[1])[:TOP_N]
    per_theme_rows[theme] = [
        {"Paper Title": paper_titles[pi], "Score": float(sc)} for (pi, sc) in sel
    ]
    all_selected_scores.extend([r["Score"] for r in per_theme_rows[theme]])

if len(all_selected_scores) == 0:
    all_selected_scores = [0.0]
global_max = max(all_selected_scores)
xmax = 1.05 if global_max <= 1.0 else max(1.05, global_max * 1.05)

# --- Plot: same style as your original (TOP_N bars per theme) ---
total_themes = len(themes)
rows = math.ceil(total_themes / COLUMNS)
fig, axs = plt.subplots(rows, COLUMNS, figsize=(FIG_W, ROW_H * rows))
axs = np.array(axs).reshape(-1) if isinstance(axs, (list, np.ndarray)) else np.array([axs])

for i, theme in enumerate(themes):
    ax = axs[i]
    data = per_theme_rows[theme]

    if len(data) == 0:
        bars = ax.barh(["No unique papers available"], [0.0], color='lightblue')
        ax.set_title(theme, fontsize=10)
        ax.set_xlim(0, xmax)
        ax.invert_yaxis()
        ax.tick_params(axis='y', labelsize=8)
        ax.text(xmax * 0.01, 0, "0.00", va='center', fontsize=8)
        continue

    titles = [r["Paper Title"] for r in data]
    scs = [r["Score"] for r in data]

    bars = ax.barh(titles, scs, color='lightblue')
    ax.set_title(theme, fontsize=10)
    ax.invert_yaxis()  # Highest on top
    ax.tick_params(axis='y', labelsize=8)
    ax.set_xlim(0, xmax)

    # Add score labels
    for j, bar in enumerate(bars):
        sc = scs[j]
        ax.text(bar.get_width() + xmax * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{sc:.2f}", va='center', fontsize=8)

# Remove any unused subplots
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])

# --- Print each theme with paper name and score ---
for theme in themes:
    print(f"\n=== {theme} ===")
    rows = per_theme_rows.get(theme, [])
    if not rows:
        print("  (No unique papers available)")
        continue
    for rank, r in enumerate(rows, start=1):
        print(f"  {rank}. {r['Paper Title']} — {r['Score']:.4f}")

    
plt.tight_layout()
plt.subplots_adjust(wspace=1, hspace=0.5, left=0.15, right=0.97, top=0.95, bottom=0.1)
plt.savefig("highest_ranking_papers_unique.png", dpi=250)
plt.show()
