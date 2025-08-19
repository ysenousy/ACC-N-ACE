import pandas as pd
import matplotlib.pyplot as plt

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
plt.title("Average Theme Scores Across Papers")
plt.xlabel("Themes")
plt.ylabel("Average Cosine Similarity Score")
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.tight_layout()
plt.savefig("average_score_per_theme.png", dpi=300)
plt.show()
