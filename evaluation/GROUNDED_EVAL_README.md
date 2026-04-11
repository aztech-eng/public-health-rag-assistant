# Grounded Offline Eval Set

This folder includes an offline benchmark focused on grounded answer quality.

## Files

- `evaluation/hpr_grounded_eval_offline.json`: grounded eval dataset
- `scripts/build_grounded_eval_set.py`: rebuilds dataset from local BM25 index + seed questions
- `scripts/evaluate_grounded_offline.py`: scores model outputs against grounded eval set without LLM judges

## Dataset schema

Top-level keys:

- `dataset_name`, `version`, `generated_at_utc`
- `index_snapshot` (BM25 snapshot metadata used during generation)
- `records` (list of evaluation items)

Each record includes:

- `id`
- `question`
- `type` (`factual`, `comparison`, `analysis`, `edge`)
- `expected_behavior` (`answer` or `abstain`)
- `reference_answer`
- `required_keywords`
- `evidence` (source/page/chunk/quote from local corpus)
- `rubric`

## Regenerate the grounded eval set

```powershell
python scripts/build_grounded_eval_set.py `
  --seed-file evaluation/hpr_eval.json `
  --index-path .rag/index.json `
  --output-file evaluation/hpr_grounded_eval_offline.json
```

## Run offline grounded evaluation

`--predictions` should point to a JSON array with `question`, `answer`, and optional `citations` fields.

Example (using existing benchmark outputs):

```powershell
python scripts/evaluate_grounded_offline.py `
  --eval-set evaluation/hpr_grounded_eval_offline.json `
  --predictions evaluation/out/llm_20/results.json `
  --output-dir evaluation/out/grounded_offline_llm_20
```

Outputs:

- `results.json` (per-question groundedness scores)
- `summary.json` (aggregated by type + overall)
- `summary.txt` (human-readable summary)
