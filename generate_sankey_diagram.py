import pandas as pd
import plotly.graph_objects as go

# === Load the CSV ===
df = pd.read_csv("classified_papers_semantic_weighted.csv")

# === Melt the dataframe to long format ===
melted = df.melt(id_vars=["Paper Title"], var_name="Theme", value_name="Score")

# === Filter only 'Directly Addressed' entries ===
filtered = melted[melted["Score"].str.strip().str.lower() == "directly addressed"]

# === Create label list ===
paper_titles = filtered["Paper Title"].unique().tolist()
themes = filtered["Theme"].unique().tolist()
labels = paper_titles + themes

# === Map labels to index ===
label_indices = {label: i for i, label in enumerate(labels)}

# === Define source, target, value for Sankey diagram ===
sources = filtered["Paper Title"].map(label_indices)
targets = filtered["Theme"].map(label_indices)
values = [1] * len(filtered)  # each link has equal weight

# === Create the Sankey Diagram ===
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=labels,
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values
    ))])

fig.update_layout(title_text="Sankey Diagram: Papers to Themes", font_size=12,
    width=2000,   
    height=1200)
fig.write_image("sankey_diagram.png", scale=10)
fig.show()
