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

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(theme_names, average_scores, marker='o', linestyle='-')

# Title and labels
plt.title("Average Theme Scores Across Papers", fontsize=16)
plt.xlabel("Themes", fontsize=14)
plt.ylabel("Average Cosine Similarity Score", fontsize=14)

# Tick label font sizes
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)

# Grid
plt.grid(True)

# Add values above each point
for i, score in enumerate(average_scores):
    plt.text(
        i, 
        score + 0.01, 
        f"{score:.3f}",
        ha='center',
        va='bottom',
        fontsize=11
    )

plt.tight_layout()
plt.savefig("average_score_per_theme.png", dpi=300)
plt.show()
