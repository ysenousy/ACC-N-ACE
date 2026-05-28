# Automated Compliance Checking in AEC: Literature Review Dataset and Visual Analysis

## Project Overview

This repository supports an academic literature review on Automated Compliance Checking (ACC) in the Architecture, Engineering, and Construction (AEC) domain. The project consolidates bibliographic records from multiple scholarly search sources, extracts textual and multimodal evidence from selected papers, classifies papers against a 10-theme analytical framework, and generates visual summaries for research interpretation.

The workflow is designed to support transparent and repeatable evidence synthesis. It combines structured search-query outputs, PDF text extraction, optical character recognition (OCR), sentence-transformer semantic similarity scoring, threshold-based thematic classification, sensitivity analysis, and publication-oriented visualisations.

## Research Scope

The analysis focuses on recent ACC-related literature and classifies papers according to the following 10 themes:

1. Multimodal Information Extraction
2. Semantic Alignment with BIM/IFC
3. Formalisation of Regulatory Text
4. Rule Representation and Reasoning
5. Integration of Ontologies and Knowledge Graphs
6. Model-Driven Compliance Intelligence
7. Explainability and Trust in AI Systems
8. Human-in-the-Loop Approaches
9. Evaluation and Benchmarking
10. Tool Development and Real-World Application

Each paper is labelled per theme as:

- `directly addressed`
- `partially addressed`
- `not addressed`

Semantic scores and representative evidence sentences are retained where available.

## Repository Structure

| Path | Description |
|---|---|
| `SearchQueries/` | Search results exported from DeepSearch, Scopus, Web of Science, and IEEE Xplore. |
| `extracted_papers/` | Extracted paper text, tables, image references, and OCR-derived text. |
| `extracted_papers/images/` | Images extracted from source PDFs for OCR and multimodal inspection. |
| `classified_papers_semantic_weighted.csv` | Main semantic classification output. |
| `classified_papers_semantic_weighted_*.csv` | Sensitivity variants for threshold or weighting experiments. |
| `sensitivity_outcomes.csv` | Summary table comparing baseline and sensitivity settings. |
| `*.png` | Generated visual outputs used for analysis and reporting. |

## Main Workflow

### 1. Consolidate Search Results

`unified_data_clean_final.py` combines search results from multiple sources into a single cleaned CSV file:

```powershell
python unified_data_clean_final.py
```

Main output:

```text
AllSources_Combined_Clean_Final.csv
```

### 2. Extract Paper Content

`pdfs_to_txt_v2.py` extracts text, tables, images, and OCR text from PDF files:

```powershell
python pdfs_to_txt_v2.py
```

Expected input:

```text
PDFs/
```

Main outputs:

```text
extracted_papers/*.txt
extracted_papers/images/*
```

### 3. Classify Papers Semantically

`classify_papers_semantic_v3.py` uses sentence-transformer embeddings to compare extracted paper sentences with predefined theme descriptions. It applies weighted section scores for main text, tables, and OCR-derived image text.

```powershell
python classify_papers_semantic_v3.py
```

Main output:

```text
classified_papers_semantic_weighted.csv
```

### 4. Run Sensitivity Analysis

`build_sensitivity_outcomes.py` compares the baseline classification against alternative CSV variants:

```powershell
python build_sensitivity_outcomes.py --baseline classified_papers_semantic_weighted.csv --files classified_papers_semantic_weighted_Stricter.csv classified_papers_semantic_weighted_Looser.csv classified_papers_semantic_weighted_Conservative.csv classified_papers_semantic_weighted_Stronger.csv --out sensitivity_outcomes.csv
```

This produces theme-stability, overlap, Jaccard similarity, and classification-count summaries.

### 5. Generate Figures

The project includes several visualisation scripts:

| Script | Output |
|---|---|
| `generate_bar_chart.py` | `theme_coverage.png` |
| `generate_heatmap.py` | `heatmap.png` |
| `generate_line_chart_themes.py` | `average_score_per_theme.png` |
| `generate_dot_matrix.py` | `paper_theme_dot_matrix.png` |
| `generate_sankey_diagram.py` | `sankey_diagram.png` |
| `generate_highest_ranking.py` | `highest_ranking_papers.png` |
| `generate_highest_ranking_unique.py` | `highest_ranking_papers_unique.png` |

Example:

```powershell
python generate_bar_chart.py
python generate_heatmap.py
python generate_line_chart_themes.py
python generate_dot_matrix.py
```

If a PNG is open in another application, Windows may prevent Python from overwriting it. Close the image viewer or document editor before rerunning the relevant script.

## Methodological Notes

The semantic classification procedure uses the `all-MiniLM-L6-v2` sentence-transformer model to compute similarity between extracted paper sentences and theme definitions. The highest weighted sentence-level similarity score per theme is retained for each paper.

The current baseline configuration uses:

```text
Direct threshold: 0.70
Partial threshold: 0.50
Text weight: 1.00
Table weight: 0.30
Image OCR weight: 0.20
```

These thresholds and weights are implemented in `classify_papers_semantic_v3.py`. Sensitivity variants are included to examine whether conclusions remain stable under stricter, looser, conservative, or stronger weighting assumptions.

## Software Requirements

The scripts are written in Python and use the following main packages:

```text
pandas
matplotlib
seaborn
plotly
sentence-transformers
torch
PyMuPDF
pdfplumber
pdfminer.six
pdf2image
pytesseract
opencv-python
kaleido
```

OCR and PDF image conversion may also require local system dependencies:

- Tesseract OCR
- Poppler for `pdf2image`

## Reproducibility

To reproduce the main analysis from source documents:

1. Place source PDFs in `PDFs/`.
2. Run `python pdfs_to_txt_v2.py`.
3. Run `python classify_papers_semantic_v3.py`.
4. Run figure-generation scripts as required.
5. Run `build_sensitivity_outcomes.py` if comparing classification variants.

For reproducible academic reporting, record the Python version, package versions, classification thresholds, and any manual inclusion or exclusion decisions used during paper screening.

## Research Outputs

The repository currently contains visual outputs for:

- Thematic coverage by addressing level.
- Heatmap of paper-theme semantic coverage scores.
- Average score per theme.
- Paper-theme direct-addressing dot matrix.
- Sankey mapping between papers and directly addressed themes.
- Highest-ranking papers per theme.
- Sensitivity outcomes across classification settings.

These outputs are intended to support interpretation of coverage patterns, methodological maturity, and research gaps across ACC literature.

## Limitations

The classification is semi-automated and depends on the quality of PDF extraction, OCR, theme definitions, threshold settings, and semantic-model behaviour. Results should therefore be treated as analytical support for expert review rather than a substitute for close reading. Manual validation remains important, particularly for papers with complex regulatory logic, domain-specific terminology, or limited extractable text.

## Citation and Use

If this repository is used in academic work, cite the underlying literature review or thesis/dissertation associated with the project, and describe the semantic classification method, thresholds, and sensitivity analysis settings used to generate the reported outputs.
