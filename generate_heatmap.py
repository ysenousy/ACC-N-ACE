import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

TITLE_SIZE = 20
AXIS_LABEL_SIZE = 16
TICK_LABEL_SIZE = 14
PAPER_LABEL_SIZE = 10
THEME_BOX_SIZE = 16
COLORBAR_SIZE = 14

# Load the CSV
df = pd.read_csv("classified_papers_semantic_weighted.csv")

# Extract only score columns
score_cols = [col for col in df.columns if "(score)" in col]
df_scores = df[score_cols]

# Extract theme names (remove " (score)" suffix)
theme_names = [col.replace(" (score)", "") for col in score_cols]

# Add paper titles as index
df_scores.index = df["Authors"]

# Generate heatmap
plt.figure(figsize=(18, 10))
ax = sns.heatmap(
    df_scores, 
    cmap="YlGnBu", 
    linewidths=0.5, 
    linecolor='gray',
    cbar_kws={'label': 'Coverage Score'}
)

# Titles and labels
plt.title("Heatmap of Thematic Coverage Scores per Paper", fontsize=TITLE_SIZE, pad=18)
plt.xlabel("Themes", fontsize=AXIS_LABEL_SIZE, labelpad=10)
plt.ylabel("Papers", fontsize=AXIS_LABEL_SIZE, labelpad=10)

# Tick labels
# Set x-axis labels to t1, t2, ..., t10
x_labels = [f"T{i+1}" for i in range(len(score_cols))]
ax.set_xticklabels(x_labels, rotation=0, ha="center", fontsize=TICK_LABEL_SIZE)
plt.yticks(fontsize=PAPER_LABEL_SIZE)

# Create legend box with theme names
legend_text = "Themes:\n" + "\n".join([f"T{i+1}: {theme_names[i]}" for i in range(len(theme_names))])
plt.figtext(0.58, 0.5, legend_text, ha='left', fontsize=THEME_BOX_SIZE, 
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='lightgray', alpha=0.95),
            verticalalignment='center', transform=plt.gcf().transFigure)

# Colorbar font sizes
cbar = ax.figure.axes[-1]
cbar.yaxis.label.set_size(COLORBAR_SIZE)     # colorbar label
cbar.tick_params(labelsize=COLORBAR_SIZE)    # colorbar tick labels

plt.tight_layout()
# Add extra space on the left to prevent Y-axis label overlap, and on the right for legend
plt.subplots_adjust(left=0.13, right=0.55)
plt.savefig("heatmap.png", dpi=300, bbox_inches="tight")
plt.show()
