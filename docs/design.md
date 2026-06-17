# A Tool-augmented agentic workflow for automated model evaluation and reporting

## Agentic design highlights

- **Dual-agent pattern:** Analyst agent synthesizes evidence; Reviewer agent validates accuracy.
- **Tool-augmented reasoning:** Agents call structured tools for dataset and overall context.
- **Conversational memory:** Shared registries keep context across agent turns.

## Key Parameters

- `MODEL_VERSION_NEW` (MLP1)
- `MODEL_VERSION_OLD` (ML)
- `OUTPUT_FOLDER`
- `MODEL_ID` (Bedrock model id)

## Winner Logic

- **clean (FP)** dataset:
  - fewer convictions wins
- **mal (FN)** dataset:
  - more convictions wins
- equal convictions:
  - same performance

## Agentic workflow

1. Discover valid datasets.
2. Compute comparison metrics and totals.
3. Register per-dataset and overall context.
4. Analyst agent generates bullet summaries using tools.
5. Reviewer agent audits and records approval feedback.
6. Build HTML sections and optional Confluence publish.
