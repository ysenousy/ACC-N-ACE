import pandas as pd
import matplotlib.pyplot as plt

# Optional: global font size
plt.rcParams.update({"font.size": 12})

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
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(theme_labels, average_scores, marker='o', linestyle='-')

# Title and labels
ax.set_title("Average Theme Scores Across Papers", fontsize=16)
ax.set_xlabel("Themes", fontsize=14)
ax.set_ylabel("Average Cosine Similarity Score", fontsize=14)

# Tick label font sizes
ax.set_xticklabels(theme_labels, rotation=0, ha='center', fontsize=12)
ax.tick_params(axis='y', labelsize=12)

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
        fontsize=11
    )

# Add legend on the right
legend_text = "Theme:\n" + "\n".join([f"{label}: {theme}" for label, theme in theme_mapping.items()])
ax.text(1.18, 0.5, legend_text, transform=ax.transAxes, fontsize=8, verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout(rect=[0, 0, 0.75, 1])
plt.subplots_adjust(left=0.12, right=0.65)
plt.savefig("average_score_per_theme.png", dpi=300)
plt.show()
