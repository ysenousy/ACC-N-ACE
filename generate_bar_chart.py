import pandas as pd
import matplotlib.pyplot as plt

# === Step 1: Load the CSV File ===
df = pd.read_csv("classified_papers_semantic_weighted.csv")

# === Step 2: Remove 'Paper Title' and keep only base 10 themes ===
df = df.drop(columns=["Paper Title"], errors="ignore")

themes_to_keep = [
    "Multimodal Information Extraction",
    "Formalisation of Regulatory Text",
    "Semantic Alignment with BIM/IFC",
    "Integration of Ontologies and Knowledge Graphs",
    "Rule Representation and Reasoning",
    "Model–Driven Compliance Intelligence",
    "Explainability and Trust in AI Systems",
    "Human-in-the-Loop Approaches",
    "Evaluation and Benchmarking",
    "Tool Development and Real-World Application"
]
df = df[[theme for theme in df.columns if theme.strip() in themes_to_keep]]

# === Step 3: Normalize labels ===
df_normalized = df.map(lambda x: str(x).strip().lower() if isinstance(x, str) else x)

# === Step 4: Count classification levels ===
theme_stats = pd.DataFrame({
    "Directly Addressed": (df_normalized == "directly addressed").sum(),
    "Partially Addressed": (df_normalized == "partially addressed").sum(),
    "Not Addressed": (df_normalized == "not addressed").sum()
})

# === Step 5: Plot Horizontal Stacked Bar Chart ===
fig, ax = plt.subplots(figsize=(12, 8))
theme_stats.plot(
    kind="barh",
    stacked=True,
    color=["mediumaquamarine", "khaki", "gold"],
    ax=ax
)

# Add title and labels
ax.set_title("Thematic Coverage of Papers", fontsize=14)
ax.set_xlabel("Number of Papers")
ax.set_ylabel("Themes")
ax.legend(title="Addressing Level", bbox_to_anchor=(1.05, 1), loc="upper left")

# === Step 6: Add numbers on bars ===
for i, (index, row) in enumerate(theme_stats.iterrows()):
    left = 0
    for category in theme_stats.columns:
        value = row[category]
        if value > 0:
            ax.text(
                left + value / 2,  # position in the middle of the segment
                i,                 # y-position
                str(value),        # text label
                va="center",
                ha="center",
                fontsize=9,
                color="black"
            )
        left += value

plt.tight_layout()
plt.savefig("theme_coverage", dpi=300)
plt.show()
