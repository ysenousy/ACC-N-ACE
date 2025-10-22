import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv("classified_papers_semantic_weighted.csv")

# Extract only score columns
score_cols = [col for col in df.columns if "(score)" in col]
df_scores = df[score_cols]

# Optional: Add paper titles as index
df_scores.index = df["Authors"]

# Generate heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(df_scores, cmap="YlGnBu", linewidths=0.5, linecolor='gray', cbar_kws={'label': 'Coverage Score'})
plt.title("Heatmap of Thematic Coverage Scores per Paper")
plt.xlabel("Themes")
plt.ylabel("Papers")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("heatmap.png", dpi=200)
plt.show()
