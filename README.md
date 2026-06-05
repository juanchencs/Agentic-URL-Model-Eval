# Agentic-URL-Model-Eval

Agentic workflow for comparing two URL/domain ML model versions across multiple datasets, with Bedrock + LangChain analysis and optional Confluence publishing.

## Project Title

**URLEval AI — Automated ML Model Comparison with AI Insights**

## What This Project Does

- Reads all valid dataset CSVs under an output folder using pattern:
  - `{DATA_SOURCE}_{TYPE}_{FLAG}.csv`
- Validates required columns:
  - `score{MODEL_VERSION_NEW}`
  - `score{MODEL_VERSION_OLD}`
- Computes conviction metrics at threshold `score >= threshold`
- Applies winner logic:
  - **FP datasets (`clean`)**: fewer convictions wins
  - **FN datasets (`mal`)**: more convictions wins
  - equal convictions: same performance
- Uses **LangChain tool/function calling** + **AWS Bedrock Converse (Claude)** to generate concise per-dataset and overall summaries
- Builds a Confluence-compatible HTML page and can publish it to a specified Confluence parent page

## Core Scripts

- `src/2models10datasets.py`
  - orchestrates scanning two model versions across input datasets and produces combined CSV outputs
- `src/ai_2models10datasets.py`
  - performs model comparison analytics, agentic LLM summary generation, and Confluence-compatible report creation/publishing
- `src/url_scan_ai.py`
  - API helper module for start/poll/download/upload flow

## Architecture & Data Workflow Diagrams

> Place your uploaded diagrams here before pushing:
>
> - `docs/images/system_architecture.png`
> - `docs/images/data_workflow.png`

### System Architecture

![System Architecture](docs/images/system_architecture.png)

### Data Workflow

![Data Workflow](docs/images/data_workflow.png)

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

Configured in-file in script `main()`:

- `MODEL_VERSION_NEW` (e.g., `"20250101"`) 20250101 is not a valid model version for privacy reasons
- `MODEL_VERSION_OLD` (e.g., `"20240101"`)  
- `OUTPUT_FOLDER` (e.g., `/home/ubuntu/efs/urlmodel/data/output_data/`)
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
- credentials file path (e.g. `****config`)

## Notes

- Confluence publishing requires valid API token and page permissions.
