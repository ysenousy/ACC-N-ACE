import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import matplotlib.pyplot as plt

# ✅ Load Excel file
file_path = r'C:\Research Work\Research Proposal Preparation\ACC-N-AEC\AllSources_Combined_Clean_Final.xlsx'
df = pd.read_excel(file_path, dtype=str)

# ✅ Combine Title + Abstract for context
df['Text'] = (df['Title'].fillna('') + ' ' + df['Abstract'].fillna('')).str.lower()

# ✅ Keyword lists for pre-filter
acc_keywords = ["compliance checking", "rule-based", "automated code", "code validation", "regulatory compliance"]
aec_keywords = ["bim", "ifc", "construction", "architecture", "engineering", "building codes"]

# ✅ Apply keyword-based filtering first
df_keyword_filtered = df[df.apply(lambda x:
    any(k in x['Text'] for k in acc_keywords) and any(k in x['Text'] for k in aec_keywords), axis=1)]

print(f"Keyword filtering reduced dataset from {len(df)} to {len(df_keyword_filtered)} papers")

# ✅ Load Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# ✅ Mixed multi-theme queries for broad coverage
queries = [
    # Core ACC
    "automated compliance checking in building regulations and codes",
    "automated regulation validation for construction projects",
    
    # AEC context
    "compliance checking for architecture engineering and construction projects",
    "building codes and standards validation in AEC domain",
    
    # BIM & IFC
    "BIM-based code compliance checking using IFC standards",
    "model-driven validation of building regulations in BIM",
    
    # NLP
    "natural language processing for regulation interpretation",
    "semantic parsing and text mining for compliance checking",
    
    # Rule-based and Ontology
    "rule-based reasoning for automated compliance in construction",
    "ontology-driven compliance checking for building codes",
    
    # Knowledge Graph
    "knowledge graph representation for building regulation validation",
    "semantic networks for compliance checking in BIM",
    
    # AI and LLM
    "large language models for rule interpretation in construction",
    "AI and machine learning for automated code compliance"
]

# ✅ Encode papers and queries
papers_embeddings = model.encode(df_keyword_filtered['Text'].tolist(), convert_to_tensor=True)
query_embeddings = model.encode(queries, convert_to_tensor=True)

# ✅ Compute cosine similarity and take max score per paper
scores_matrix = util.cos_sim(query_embeddings, papers_embeddings)
max_scores, best_query_idx = torch.max(scores_matrix, dim=0)  # Best score and query index

# ✅ Add scores and matched query to DataFrame
df_keyword_filtered['Relevance_Score'] = max_scores.cpu().numpy()
df_keyword_filtered['Best_Matched_Query'] = [queries[i] for i in best_query_idx.cpu().numpy()]

# ✅ Plot distribution to choose threshold
plt.hist(df_keyword_filtered['Relevance_Score'], bins=30, color='blue', alpha=0.7)
plt.title('Relevance Score Distribution')
plt.xlabel('Score')
plt.ylabel('Number of Papers')
plt.show()

# ✅ Apply threshold (adjust based on histogram)
threshold = 0.55
filtered_df = df_keyword_filtered[df_keyword_filtered['Relevance_Score'] >= threshold]
filtered_df = filtered_df.sort_values(by='Relevance_Score', ascending=False)

print(f"After semantic filtering, {len(filtered_df)} papers remain")

# ✅ Save final output
output_file = r'C:\Research Work\Research Proposal Preparation\ACC-N-AEC\Filtered_ACC_AEC_Hybrid_Final.xlsx'
filtered_df.to_excel(output_file, index=False)

print(f"✅ Hybrid filtering completed. Saved to: {output_file}")
