import pandas as pd
import matplotlib.pyplot as plt

TITLE_SIZE = 20
AXIS_LABEL_SIZE = 16
TICK_LABEL_SIZE = 14
LEGEND_SIZE = 14
THEME_BOX_SIZE = 16
BAR_LABEL_SIZE = 13

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
theme_labels = [f"T{i + 1}" for i in range(len(df.columns))]
theme_mapping = dict(zip(theme_labels, df.columns))

# === Step 3: Normalize labels ===
df_normalized = df.map(lambda x: str(x).strip().lower() if isinstance(x, str) else x)

# === Step 4: Count classification levels ===
theme_stats = pd.DataFrame({
    "Directly Addressed": (df_normalized == "directly addressed").sum(),
    "Partially Addressed": (df_normalized == "partially addressed").sum(),
    "Not Addressed": (df_normalized == "not addressed").sum()
})

# === Step 5: Plot Horizontal Stacked Bar Chart ===
fig, ax = plt.subplots(figsize=(16, 10))
theme_stats.plot(
    kind="barh",
    stacked=True,
    color=["mediumaquamarine", "khaki", "gold"],
    ax=ax
)

# Add title and labels
ax.set_title("Thematic Coverage of Papers", fontsize=TITLE_SIZE, pad=18)
ax.set_xlabel("Number of Papers", fontsize=AXIS_LABEL_SIZE, labelpad=10)
ax.set_ylabel("Themes", fontsize=AXIS_LABEL_SIZE, labelpad=10)

# Tick label font sizes
ax.set_yticklabels(theme_labels)
ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)

# Legend
ax.legend(
    title="Addressing Level",
    fontsize=LEGEND_SIZE,
    title_fontsize=LEGEND_SIZE,
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)

# Theme key box
theme_text = "Themes:\n" + "\n".join(
    [f"{label}: {theme}" for label, theme in theme_mapping.items()]
)
ax.text(
    1.05,
    0.55,
    theme_text,
    transform=ax.transAxes,
    fontsize=THEME_BOX_SIZE,
    va="top",
    ha="left",
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="lightgray", alpha=0.95)
)

# === Step 6: Add numbers on bars ===
for i, (index, row) in enumerate(theme_stats.iterrows()):
    left = 0
    for category in theme_stats.columns:
        value = row[category]
        if value > 0:
            ax.text(
                left + value / 2,
                i,
                str(value),
                va="center",
                ha="center",
                fontsize=BAR_LABEL_SIZE,
                color="black"
            )
        left += value

plt.tight_layout()
plt.savefig("theme_coverage.png", dpi=300, bbox_inches="tight")
plt.show()
