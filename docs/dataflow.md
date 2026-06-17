# A Tool-augmented agentic workflow for automated model evaluation and reporting

## Agentic pipeline

1. Read `OUTPUT_FOLDER` and build `INPUT_LIST`
2. Parse metadata from filename:
   - `DATA_SOURCE`, `TYPE`, `FLAG`
3. For each dataset:
   - load CSV
   - compute confusion-style conviction counts:
     - BOTH NOT CONVICTED
     - BOTH CONVICTED
     - ONLY ML
     - ONLY MLP1
     - TOTAL CONVICTED ML
     - TOTAL CONVICTED MLP1
   - compute percentages
   - decide winner
   - generate LLM summary (max 4 bullets)
4. Build final report:
   - overall table
   - dataset sections
   - optional Confluence publish

## Diagram

![Agentic dataflow](images/dataflow.svg)
