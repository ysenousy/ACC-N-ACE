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
Extracting regulatory data from text, images, tables, diagrams using NLP and computer vision.
Keywords: Natural Language Processing, NLP, multimodal extraction, OCR, table parsing, diagram understanding, layout analysis, PDF-to-structured, document AI, computer vision in regulations, visual document parsing, multimodal NLP, image-based rules, tabular data extraction, document layout modeling, diagram OCR, vision-language models, image-text alignment, semantic segmentation.
""",

    "Formalisation of Regulatory Text": """
Transforming legal/regulatory text into structured, machine-readable formats via parsing, modeling, and logic encoding.
Keywords: regulatory formalization, rule extraction, logic encoding, syntax trees, NLP parsing, legal text processing, structured regulations, compliance logic, tokenization, Gherkin, T4R, regulatory modeling, structured rules, legal knowledge extraction, legal informatics, machine-readable law, information extraction, legal document modeling, rule authoring, rule templates, plain-language rules, regulation updates, version control, regulatory change management, delta detection, evolving rules.
""",

    "Semantic Alignment with BIM/IFC": """
Mapping regulatory requirements to BIM objects using IFC and buildingSMART standards.
Keywords: semantic alignment, IFC mapping, BIM linkage, BIM compliance, rule-to-model alignment, BIM ontology, IFC schema, object-property linking, geometry validation, model interpretation, model checking, digital twin regulation, Level of Detail (LOD), spatial reasoning, BIM data mapping, IDS, Information Delivery Specification, IFC validation rules, information delivery requirements, bSDD, buildingSMART Data Dictionary, MVD, Model View Definition, IDM, Information Delivery Manual, gbXML, CityGML, ISO 19650, BIM interoperability, non-IFC standards, BIM data exchange, information delivery standards.
""",

    "Integration of Ontologies and Knowledge Graphs": """
Using ontologies and knowledge graphs to model regulatory concepts, relations, and rule hierarchies.
Keywords: knowledge graph, ontology integration, regulatory ontology, semantic web, RDF triples, OWL classes, SPARQL queries, linked data, graph-based modeling, domain semantics, ontology-driven reasoning, concept hierarchies, data interlinking, bSDD, SKOS, SBVR, business vocabulary and rules, graph enrichment, semantic integration.
""",

    "Rule Representation and Reasoning": """
Expressing compliance rules with formal languages and enabling inference.
Keywords: rule formalization, rule-based reasoning, SHACL, SWRL, RDF, OWL, constraint validation, logic programming, inference engine, compliance engine, semantic rules, rule sets, Prolog, logic schema, automated rule evaluation, SBVR, business rules modeling, rule engine, semantic reasoning, ontology reasoner.
""",

    "Model–Driven Compliance Intelligence": """
Applying LLMs and Retrieval-Augmented Generation for compliance understanding and knowledge augmentation.
Keywords: LLM, GPT, BERT, ChatGPT, RAG, Retrieval-Augmented Generation, prompt engineering, zero-shot reasoning, few-shot prompting, fine-tuned transformer, foundation models, large models for regulation, compliance generation, text-to-rule, domain-adapted models, knowledge-augmented generation, semantic retrieval, RAG pipelines.
""",

    "Explainability and Trust in AI Systems": """
Ensuring AI compliance tools offer transparent, interpretable explanations for trust and auditability.
Keywords: explainable AI, XAI, model transparency, interpretability, human trust, decision explanation, explainable compliance, rationale generation, rule traceability, visual explanation, regulatory justification, compliance explanation engine, model auditability, human-centered AI, explanation interface, justificatory models, AI bias mitigation, data privacy, regulatory ethics, human-centered assurance, privacy-preserving compliance.
""",

    "Human-in-the-Loop Approaches": """
Incorporating human experts into compliance pipelines for supervision and refinement.
Keywords: HITL, hybrid AI-human, interactive compliance, manual override, expert validation, user-in-the-loop, participatory compliance checking, semi-automated verification, human feedback loop, collaborative AI, active learning, expert-in-the-loop, human review layer, crowdsourced feedback, human oversight.
""",

    "Evaluation and Benchmarking": """
Validating ACC systems with benchmark datasets, metrics, and comparative performance testing.
Keywords: compliance evaluation, benchmark datasets, test suite, precision-recall, F1 score, validation framework, model comparison, compliance metrics, error analysis, test-driven compliance, reproducible testing, validation dataset, performance comparison, evaluation protocol, ground-truth dataset, ACC benchmarking.
""",

    "Tool Development and Real-World Application": """
Developing and deploying ACC tools within real-world AEC workflows and systems.
Keywords: compliance tool, BIM plugin, IFC validation tool, digital permitting, software deployment, commercial integration, regulatory automation system, compliance prototype, rule engine, industry application, field validation, smart permit platform, pilot project, AEC integration, architectural tooling, construction automation, ACC software.
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
