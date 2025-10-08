import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# === Configuration ===
DIRECT_THRESHOLD = 0.70
PARTIAL_THRESHOLD = 0.50
MIN_SENTENCE_WORDS = 8

# Weights for each section
WEIGHTS = {
    "text": 1.0,       # main body
    "tables": 0.3,     # extracted tables
    "images": 0.2      # OCR from image content
}

# === Themes ===
theme_descriptions = {
    "Multimodal Information Extraction": """
Extracting regulatory data from text, images, tables, and diagrams using NLP and computer vision, then structuring it for downstream reasoning and compliance checking.
Keywords: Natural Language Processing, NLP, multimodal extraction, document AI, OCR, table extraction, diagram understanding, layout analysis, visual document parsing, PDF to structured, document layout modeling, semantic parsing, concept and parameter capture, rule mining from tables, multimodal NLP, vision language models, image text alignment, semantic segmentation, ontology guided computer vision, knowledge graphs, NLP and CV fusion.
""",

    "Formalisation of Regulatory Text": """
Building regulations are being transformed from prose and tables to precise, logic-ready structures by decomposing requirements, capturing parameters and thresholds, modeling applicability conditions and exceptions, resolving cross-references and dependencies, and encoding them in a traceable logic representation with tree and graph views for automated compliance reasoning.
Keywords: requirement decomposition, clause structuring, logic-based representation, predicate/production rules, tree/graph visualisation, parameter capture, threshold extraction, applicability conditions, exception modelling, dependency modelling, cross-reference resolution, rule atomization, source traceability, computable rules, reasoning-ready constraints.
""",

    "Semantic Alignment with BIM/IFC": """
Establishing clear and computable links between regulatory requirements and IFC based BIM data by aligning rules with specific entities, properties, and relationships, enabling accurate querying and automated compliance checking through schema mapping, ontology integration, and BIM aware rule execution.
Keywords: semantic alignment, IFC mapping, rule to model alignment, entity and property mapping, property set (Pset) alignment, BIMQL queries, constraint to IFC property mapping, rule to BIM binding, BPMN and DMN with BIM, compliance oriented object model, IFC schema extension, ontology to IFC alignment, spatial structure and topology, traceability to model elements, model checking, BIM interoperability.
""",

    "Integration of Ontologies and Knowledge Graphs": """
Ontologies and knowledge graphs used to formalize regulatory concepts, domain entities, and their interactions, allowing for semantic reasoning, rule-based compliance verification, and data integration across multiple sources such as BIM, sensors, and computer vision. These structures enable automated inference, consistency checking using formal languages like as OWL, SWRL, and SPARQL.  
Keywords: ontology modeling, knowledge graph, semantic reasoning, rule-based compliance checking, ontology-driven integration, OWL, SWRL, RDF, SPARQL queries, semantic interoperability, domain semantics, hazard detection, ontology-BIM integration, visual-semantic reasoning, logic-based rule encoding, semantic annotation, multi-source data fusion.
""",

    "Rule Representation and Reasoning": """
Creating and implementing formal representations of regulatory requirements in logic-based, semantic, or visual languages to facilitate automated compliance reasoning. These approaches include first-order logic clauses, rule trees, visual rule languages, decision models (e.g., DMN), and semantic rule formats (e.g., RuleML), all of which enable uniform, visible, and executable rule assessment within BIM and regulatory systems.
Keywords: rule representation, logic-based modeling, first-order logic, logic clauses, tree-based rule structures, visual rule language, DMN, BPMN, semantic rule encoding, RuleML, rule classification, rule schema, logic programming, compliance reasoning, rule evaluation, reasoning engine.
""",

    "Model–Driven Compliance Intelligence": """
Using large language models, deep learning, and retrieval-augmented generation to automate compliance understanding, rule extraction, and regulatory interpretation. These approaches combine semantic retrieval, ontology grounding, and few-shot learning to improve adaptability, eliminate manual rule authoring, and enable domain-specific compliance reasoning in BIM and workflows.
Keywords: large language models, LLM, GPT, BERT, ChatGPT, retrieval-augmented generation, RAG, prompt engineering, few-shot learning, one-shot learning, compliance automation, text-to-rule generation, deep learning pre-classification, ontology-enhanced LLM, semantic retrieval, vector embedding, compliance AI pipeline, fine-tuned transformer, domain adaptation, hybrid AI frameworks.
""",

    "Explainability and Trust in AI Systems": """
Building compliance checking systems in a way that offer interpretable, user-understandable outputs to help develop trust with the practitioners, auditors, and regulators. This involves application of visual logic models, transparent rule structures, and intuitive representations that allow users to trace rule decisions back to source regulations and assess system behaviour.
Keywords: explainable compliance, tree-based visualization, user-understandable logic, interpretable rules, regulatory traceability, transparent reasoning, compliance explainability, trust in AI, AI-assisted decision-making, visual logic modeling, rule transparency, system accountability.
""",

    "Human-in-the-Loop Approaches": """
Integrating human expertise into automated compliance systems to support interpretation, validation, and decision-making. These approaches ensure adaptability, trust, and accountability by involving users in system development, feedback loops, and oversight.
Keywords: human-in-the-loop, expert validation, hybrid AI-human systems, participatory compliance checking, semi-automated verification, human oversight, stakeholder collaboration, manual override, user-in-the-loop, feedback loop, human-assisted decision-making, domain expert review, adaptive compliance systems.
""",

    "Evaluation and Benchmarking": """
Evaluating the performance, accuracy, usability, and readiness of automated compliance checking systems through comparative studies, reviews, and standardized metrics. 
Keywords: benchmarking, evaluation metrics, performance assessment, system validation, comparative analysis, accuracy, precision, recall, F1-score, rule coverage, tool maturity, scientometric review, usability testing, adoption readiness, validation frameworks, review methodology, evaluation criteria, compliance testing metrics.
""",

    "Tool Development and Real-World Application": """
Designing, implementing, and validating practical software systems for automated compliance checking (ACC) within real-world environments.
Keywords: compliance tool, BIM integration, IFC-based checking, domain-specific language, tool validation, automated BIM checking, knowledge-driven tool, generative design, workflow automation, pilot project, real-world implementation, case study evaluation, ACC software development.
"""
}

# === Load Sentence Embedding Model ===
model = SentenceTransformer("all-MiniLM-L6-v2")
theme_embeddings = {k: model.encode(v, convert_to_tensor=True) for k, v in theme_descriptions.items()}

# === Extract sentences ===
def get_sentences(text):
    return [s.strip() for s in text.split('.') if len(s.strip().split()) >= MIN_SENTENCE_WORDS]

# === Fix: Extract section between known titles ===
def extract_section(content, section_name):
    start_marker = f"==== {section_name.upper()} ===="
    start_idx = content.find(start_marker)
    if start_idx == -1:
        return ""

    next_markers = [f"==== {k.upper()} ====" for k in WEIGHTS.keys() if k.lower() != section_name.lower()]
    next_markers.sort(key=lambda x: content.find(x) if content.find(x) > start_idx else float('inf'))

    end_idx = min([content.find(marker) for marker in next_markers if content.find(marker) > start_idx] + [len(content)])
    return content[start_idx + len(start_marker):end_idx].strip()

# === Analyze a single paper ===
def analyze_paper(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    sections = {
        "text": extract_section(content, "text"),
        "tables": extract_section(content, "tables"),
        "images": extract_section(content, "ocr text from images")
    }

    max_scores = {theme: 0.0 for theme in theme_descriptions}
    top_sentences = {theme: "" for theme in theme_descriptions}

    for section, text in sections.items():
        weight = WEIGHTS.get(section, 1.0)
        for sentence in get_sentences(text):
            emb = model.encode(sentence, convert_to_tensor=True)
            for theme, theme_emb in theme_embeddings.items():
                sim = util.pytorch_cos_sim(emb, theme_emb).item()
                weighted_sim = sim * weight
                if weighted_sim > max_scores[theme]:
                    max_scores[theme] = weighted_sim
                    top_sentences[theme] = sentence

    title = os.path.basename(filepath).replace(".txt", "")
    result = {"Paper Title": title}
    for theme in theme_descriptions:
        score = max_scores[theme]
        label = (
            "directly addressed" if score >= DIRECT_THRESHOLD
            else "partially addressed" if score >= PARTIAL_THRESHOLD
            else "not addressed"
        )
        result[f"{theme}"] = label
        result[f"{theme} (score)"] = round(score, 3)
        result[f"{theme} (sentence)"] = top_sentences[theme]

    return result

# === Process All Files ===
papers_dir = "extracted_papers"
results = []
for file in os.listdir(papers_dir):
    if file.endswith(".txt"):
        print(f"Analyzing: {file}")
        full_path = os.path.join(papers_dir, file)
        results.append(analyze_paper(full_path))

# === Save Output ===
df = pd.DataFrame(results)
df.to_csv("classified_papers_semantic_weighted.csv", index=False)
print("Results saved to classified_papers_semantic_weighted.csv")
