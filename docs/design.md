# Design

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

## Agentic Workflow

1. Discover valid datasets
2. Compute table metrics
3. Register per-dataset and overall context
4. Run LLM analysis with tool/function calling
5. Build HTML sections and optional Confluence publish
