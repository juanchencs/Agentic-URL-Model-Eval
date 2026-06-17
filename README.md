# A Tool-augmented agentic workflow for automated model evaluation and reporting

This repository provides a dual-agent evaluation pipeline that compares two URL/domain ML models across multiple datasets and produces reviewed, publishable reports. The workflow centers on an **Analyst agent** (summaries) and a **Reviewer agent** (quality gate).

## Why an agentic workflow

- **Tool selection:** the analyst invokes structured tools for dataset and overall context.
- **Conversational memory:** shared registries keep dataset-level context across steps.
- **Dual-agent design:** the reviewer validates the analyst output for accuracy and format.

## Workflow overview

1. Discover dataset files named `{DATA_SOURCE}_{TYPE}_{FLAG}.csv`.
2. Validate required score columns for both model versions.
3. Compute per-dataset comparison tables (counts + percentages).
4. **Analyst agent** generates evidence-backed bullet summaries via tool calls.
5. **Reviewer agent** audits the summaries and records approval feedback.
6. Build Confluence-ready HTML and optional report artifacts.

## Core scripts

- `src/2models10datasets.py`: scans two model versions across input datasets.
- `src/ai_2models10datasets.py`: comparison analytics + agent summaries + report output.
- `src/url_scan_ai.py`: API helper for scan orchestration.

## Dataflow

![Agentic dataflow](docs/images/dataflow.svg)

## Repository Structure

```text
Agentic-URL-Model-Eval/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── src/
│   ├── 2models10datasets.py
│   └── ai_2models10datasets.py
│   └── url_scan_ai.py
├── tests/
│   ├── vt_domain_clean.txt
│   ├── vt_url_clean.txt
│   ├── vt_domain_mal.txt
│   ├── vt_url_mal.txt
│   └── test_filename_pattern.py
└── docs/
    ├── architecture.md
    ├── design.md
    ├── dataflow.md
    └── images/
```

## Parameters

Configured in `main()`:

- `MODEL_VERSION_NEW` (e.g., `"123456"`)
- `MODEL_VERSION_OLD` (e.g., `"654321"`)
- `OUTPUT_FOLDER` (e.g., `/path/to/output_data/`)
- `MODEL_ID` (e.g., `claude-opus`)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage Examples

### 1) Run two-model scanning + merged outputs

```bash
python3 src/2models10datasets.py
```

### 2) Run agentic analysis + local Confluence-compatible HTML

```bash
python3 src/ai_2models10datasets.py
```

### 3) Enable Confluence publishing

Edit `main(CONFLUENCE_PUBLISH=...)` in `src/ai_2models10datasets.py` and set:

- `CONFLUENCE_PUBLISH=True`
- parent page id / space / base URL

## Notes

- Confluence publishing requires valid API token and page permissions.
