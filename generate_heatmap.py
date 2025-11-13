import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Optional: global font size
plt.rcParams.update({"font.size": 10})

# Load the CSV
df = pd.read_csv("classified_papers_semantic_weighted.csv")

# Extract only score columns
score_cols = [col for col in df.columns if "(score)" in col]
df_scores = df[score_cols]

# Add paper titles as index
df_scores.index = df["Authors"]

# Generate heatmap
plt.figure(figsize=(15, 10))
ax = sns.heatmap(
    df_scores, 
    cmap="YlGnBu", 
    linewidths=0.5, 
    linecolor='gray',
    cbar_kws={'label': 'Coverage Score'}
)

# Titles and labels
plt.title("Heatmap of Thematic Coverage Scores per Paper", fontsize=16)
plt.xlabel("Themes", fontsize=14)
plt.ylabel("Papers", fontsize=14)

# Tick labels
plt.xticks(rotation=45, ha="right", fontsize=12)
plt.yticks(fontsize=10)

# Colorbar font sizes
cbar = ax.figure.axes[-1]
cbar.yaxis.label.set_size(12)     # colorbar label
cbar.tick_params(labelsize=12)    # colorbar tick labels

plt.tight_layout()
plt.savefig("heatmap.png", dpi=200)
plt.show()
