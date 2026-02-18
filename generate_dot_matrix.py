import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = "classified_papers_semantic_weighted.csv"
DIRECT_LABEL = "directly addressed"

TOP_N_THEMES = 10   # reduce themes to keep plot readable (set None to keep all)
N_PAPERS = 30       # sample size for papers in the plot (set None to keep all)

df = pd.read_csv(CSV_PATH)
theme_cols = [c for c in df.columns if c != "Authors" and not c.endswith(("(score)", "(sentence)"))]

# Build binary matrix: 1 if directly addressed
norm = df[theme_cols].astype(str).apply(lambda s: s.str.strip().str.lower())
A = (norm == DIRECT_LABEL).astype(int)
A.index = df["Authors"].astype(str)

# Keep top themes by coverage (optional) - maintain original order
if TOP_N_THEMES is not None:
    top_themes_ranked = A.sum(axis=0).sort_values(ascending=False).head(TOP_N_THEMES).index
    # Preserve original order from CSV
    top_themes = [t for t in A.columns if t in top_themes_ranked]
    A = A[top_themes]

# Sample papers for visualization (recommended: top by coverage)
if N_PAPERS is not None:
    paper_counts = A.sum(axis=1).sort_values(ascending=False)
    A = A.loc[paper_counts.head(N_PAPERS).index]

# Plot dot matrix
rows, cols = np.where(A.values == 1)

# Create theme labels: T1, T2, etc.
theme_labels = [f"T{i+1}" for i in range(A.shape[1])]
theme_mapping = {f"T{i+1}": theme for i, theme in enumerate(A.columns)}

fig, ax = plt.subplots(figsize=(14, 0.28 * len(A) + 2))
ax.scatter(cols, rows, s=35)

ax.set_xticks(range(A.shape[1]))
ax.set_xticklabels(theme_labels, fontsize=9)
ax.set_yticks(range(A.shape[0]))
ax.set_yticklabels(A.index, fontsize=8)

ax.set_xlabel("Themes")
ax.set_ylabel("Papers (sampled for visualization)")
ax.set_title("Paper–Theme Dot Matrix (Directly Addressed)")
ax.grid(axis="x", linestyle="--", alpha=0.3)

# Add legend on the right
legend_text = "Themes:\n" + "\n".join([f"{label}: {theme}" for label, theme in theme_mapping.items()])
ax.text(1.02, 0.5, legend_text, transform=ax.transAxes, fontsize=8, verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()

plt.savefig("paper_theme_dot_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved: paper_theme_dot_matrix.png")
