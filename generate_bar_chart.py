import pandas as pd
import matplotlib.pyplot as plt

# === Step 1: Load the CSV File ===
df = pd.read_csv("classified_papers_semantic_weighted.csv")

# === Step 2: Remove 'Paper Title' and keep only base 10 themes ===
df = df.drop(columns=["Paper Title"], errors="ignore")

# Filter to keep only rows (columns) that match the 10 base themes exactly
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
plt.figure(figsize=(12, 8))
theme_stats.plot(
    kind="barh",
    stacked=True,
    color=["mediumaquamarine", "khaki", "gold"],
    figsize=(12, 8)
)
plt.title("Thematic Coverage of Papers", fontsize=14)
plt.xlabel("Number of Papers")
plt.ylabel("Themes")
plt.legend(title="Addressing Level", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("theme_coverage.png", dpi=300)
plt.show()
