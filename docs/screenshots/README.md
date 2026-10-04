# Screenshots

Taken against the **synthetic sample corpus** in `data/sample/`, never a real
index — a screenshot of real data would publish actual quiz titles, document
ids and question content.

| File | Shows |
|------|-------|
| `ui-generate.png`  | The form plus a generated quiz |
| `ui-retrieval.png` | The debug panel: retrieved chunks with distances and timings |

Reproduce from a checkout of THIS repository. Never run these in a working
copy that holds a real index — `vector_store.reset_on_build: true` deletes the
persist directory before writing.

```bash
python -m quiz_generator.data.ingest --input data/sample/quizzes-sample-raw.json \
    --scope configs/scope.yaml \
    --output data/sample/interim/flat.jsonl \
    --stats data/sample/interim/flat_stats.json
python -m quiz_generator.data.normalize --input data/sample/interim/flat.jsonl \
    --output data/sample/interim/normalized.jsonl \
    --stats data/sample/interim/normalized_stats.json
python -m quiz_generator.data.build_index_text --input data/sample/interim/normalized.jsonl \
    --output data/sample/processed/ready.jsonl \
    --stats data/sample/processed/ready_stats.json
python -m quiz_generator.indexing.build --input data/sample/processed/ready.jsonl
python -m quiz_generator.api          # http://localhost:8000/ui
```
