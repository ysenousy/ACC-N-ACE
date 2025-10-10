import pandas as pd
import matplotlib.pyplot as plt
import math

# === Configuration ===
TOP_N = 6  # Number of top papers per theme
COLUMNS = 2  # Number of diagrams per row

# Load data
df = pd.read_csv("classified_papers_semantic_weighted.csv")

# Extract themes (ending in ' (score)')
score_cols = [col for col in df.columns if col.endswith(" (score)")]
themes = [col.replace(" (score)", "") for col in score_cols]

# Prepare grid
total_themes = len(themes)
rows = math.ceil(total_themes / COLUMNS)
fig, axs = plt.subplots(rows, COLUMNS, figsize=(19, 4 * rows))
axs = axs.flatten()

# Plot each theme
for i, theme in enumerate(themes):
    score_col = f"{theme} (score)"
    top_df = df[['Paper Title', score_col]].sort_values(by=score_col, ascending=False).head(TOP_N)
    print(top_df)
    ax = axs[i]
    bars = ax.barh(top_df['Paper Title'], top_df[score_col], color='lightblue')
    ax.set_title(theme, fontsize=10)
    ax.invert_yaxis()  # Highest on top
    ax.tick_params(axis='y', labelsize=8)
    ax.set_xlim(0, 1.05)  # Scale from 0 to 1 for clarity

    # Add score labels beside each bar
    for j, bar in enumerate(bars):
        score = top_df[score_col].iloc[j]
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score:.2f}", va='center', fontsize=8)

# Remove unused subplots (if any)
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
plt.subplots_adjust(wspace=3.5, hspace=0.5, left=0.38, right=0.97, top=0.95, bottom=0.1)
plt.savefig("highest_ranking_papers.png", dpi=250)
plt.show()
