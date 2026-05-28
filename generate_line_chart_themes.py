import pandas as pd
import matplotlib.pyplot as plt

TITLE_SIZE = 20
AXIS_LABEL_SIZE = 16
TICK_LABEL_SIZE = 14
THEME_BOX_SIZE = 16
POINT_LABEL_SIZE = 13

# Load the CSV file
df = pd.read_csv("classified_papers_semantic_weighted.csv")

# Extract only the score columns (those that contain '(score)')
score_columns = [col for col in df.columns if '(score)' in col]

# Compute the average score for each theme
average_scores = df[score_columns].mean()

# Clean column names for display (remove ' (score)' suffix)
theme_names = [col.replace(' (score)', '') for col in score_columns]

# Create T1, T2, ... labels
theme_labels = [f"T{i+1}" for i in range(len(theme_names))]
theme_mapping = {f"T{i+1}": theme for i, theme in enumerate(theme_names)}

# Plotting
fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(theme_labels, average_scores, marker='o', linestyle='-')

# Title and labels
ax.set_title("Average Theme Scores Across Papers", fontsize=TITLE_SIZE, pad=18)
ax.set_xlabel("Themes", fontsize=AXIS_LABEL_SIZE, labelpad=10)
ax.set_ylabel("Average Cosine Similarity Score", fontsize=AXIS_LABEL_SIZE, labelpad=10)

# Tick label font sizes
ax.set_xticks(range(len(theme_labels)))
ax.set_xticklabels(theme_labels, rotation=0, ha='center', fontsize=TICK_LABEL_SIZE)
ax.tick_params(axis='y', labelsize=TICK_LABEL_SIZE)

# Grid
ax.grid(True)

# Add values above each point
for i, score in enumerate(average_scores):
    ax.text(
        i, 
        score + 0.01, 
        f"{score:.3f}",
        ha='center',
        va='bottom',
        fontsize=POINT_LABEL_SIZE
    )

# Add legend on the right
legend_text = "Themes:\n" + "\n".join([f"{label}: {theme}" for label, theme in theme_mapping.items()])
ax.text(
    1.05,
    0.5,
    legend_text,
    transform=ax.transAxes,
    fontsize=THEME_BOX_SIZE,
    verticalalignment='center',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='lightgray', alpha=0.95)
)

plt.tight_layout()
plt.subplots_adjust(left=0.12, right=0.58)
plt.savefig("average_score_per_theme.png", dpi=300, bbox_inches="tight")
plt.show()
