#!/usr/bin/env bash
# run_local.sh — bring quiz-ai up from a cold checkout, natively (no Docker).
#
# Idempotent: every stage is skipped if its output already exists.
# Pass --rebuild to force the whole data pipeline to run again.
#
#   ./run_local.sh              # preflight + build what's missing + serve
#   ./run_local.sh --check      # preflight only, change nothing
#   ./run_local.sh --rebuild    # force full ETL + reindex, then serve
#   ./run_local.sh --no-serve   # build everything, don't start the API
#
# Why this exists: the README's rebuild sequence is missing `--scope` on
# ingest and omits the build_index_text stage entirely, so following it
# produces an empty index. This script runs the correct sequence.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="$(pwd)"

REBUILD=0; SERVE=1; CHECK_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --rebuild)  REBUILD=1 ;;
    --no-serve) SERVE=0 ;;
    --check)    CHECK_ONLY=1; SERVE=0 ;;
    *) echo "unknown flag: $arg"; exit 2 ;;
  esac
done

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
die()  { printf '  \033[31m✗\033[0m %s\n' "$*"; exit 1; }

RAW=data/raw/quizzes-raw-data.json
FLAT=data/interim/flat.jsonl
NORM=data/interim/normalized.jsonl
READY=data/processed/ready_phase1.jsonl
CHROMA=data/vector_store/chroma_db_phase1
SUMMARY=data/vector_store/build_summary.json

# ---------------------------------------------------------------- 1. python
bold "1. Python"
PY=""
for c in python3.12 python3.11 python3; do
  if command -v "$c" >/dev/null 2>&1; then
    v=$("$c" -c 'import sys;print("%d.%d"%sys.version_info[:2])')
    major=${v%%.*}; minor=${v##*.}
    if [ "$major" -eq 3 ] && [ "$minor" -ge 11 ]; then PY="$c"; break; fi
  fi
done
[ -n "$PY" ] || die "need Python 3.11+ (found: $(python3 -V 2>&1 || echo none))"
ok "$PY ($($PY -V 2>&1))"

# ------------------------------------------------------------------ 2. venv
bold "2. Virtualenv"
if [ ! -d .venv ]; then
  [ "$CHECK_ONLY" -eq 1 ] && warn ".venv missing (would be created)" || {
    "$PY" -m venv .venv; ok "created .venv"; }
else
  ok ".venv exists"
fi
if [ -d .venv ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  if [ "$CHECK_ONLY" -eq 0 ]; then
    if ! python -c 'import fastapi, chromadb, sentence_transformers' 2>/dev/null; then
      echo "  installing requirements (this pulls torch, ~2-3 GB, be patient)…"
      pip install --quiet --upgrade pip
      pip install --quiet -r requirements.txt
      ok "dependencies installed"
    else
      ok "dependencies present"
    fi
  fi
fi

# ------------------------------------------------------------------- 3. env
bold "3. Environment / API key"
if [ -f .env ]; then
  set -a; # shellcheck disable=SC1091
  source .env; set +a
  ok ".env loaded into the environment"
else
  warn "no .env file at $ROOT/.env"
fi

PROVIDER=$(python - <<'PY' 2>/dev/null || echo unknown
import yaml
print((yaml.safe_load(open("configs/models.yaml")) or {}).get("llm", {}).get("provider", "?"))
PY
)
MODEL=$(python - <<'PY' 2>/dev/null || echo unknown
import yaml
print((yaml.safe_load(open("configs/models.yaml")) or {}).get("llm", {}).get("model", "?"))
PY
)
echo "  configs/models.yaml → provider=$PROVIDER model=$MODEL"
case "$PROVIDER" in
  gemini) [ -n "${GEMINI_API_KEY:-}" ] && ok "GEMINI_API_KEY is set" \
            || die "GEMINI_API_KEY not set. echo 'GEMINI_API_KEY=...' > .env  (key: https://aistudio.google.com/app/apikey)" ;;
  groq)   [ -n "${GROQ_API_KEY:-}" ]   && ok "GROQ_API_KEY is set" \
            || die "GROQ_API_KEY not set. echo 'GROQ_API_KEY=...' > .env" ;;
  ollama) ok "ollama provider — make sure 'ollama serve' is running and the model is pulled" ;;
  *)      warn "could not read llm.provider from configs/models.yaml" ;;
esac

# Keep the run log out of the container-only default path (/app/logs/...).
export RUNS_LOG_PATH="${RUNS_LOG_PATH:-$ROOT/logs/runs.jsonl}"
mkdir -p "$(dirname "$RUNS_LOG_PATH")"
ok "RUNS_LOG_PATH=$RUNS_LOG_PATH"

# ------------------------------------------------------------------ 4. data
bold "4. Data artifacts"
have_index=0
[ -d "$CHROMA" ] && [ -f "$SUMMARY" ] && [ -f "$READY" ] && have_index=1

if [ "$have_index" -eq 1 ] && [ "$REBUILD" -eq 0 ]; then
  ok "index + payload present — nothing to build"
  ok "$(wc -l < "$READY" | tr -d ' ') rows in $READY"
elif [ ! -f "$RAW" ] && [ "$have_index" -eq 0 ]; then
  echo
  die "BLOCKED — no data.
     Missing both of:
       a) $RAW          (the raw corpus, ~271 MB, gitignored)
       b) $READY + $CHROMA/ + $SUMMARY   (the prebuilt artifacts)
     You need one of them. Restore from your backup / the machine you
     deployed from / whoever delivered the dataset, then re-run this script."
else
  [ "$CHECK_ONLY" -eq 1 ] && { warn "data pipeline would run here"; exit 0; }
  [ -f "$RAW" ] || die "--rebuild requested but $RAW is missing"

  bold "   building (this takes ~10-15 min; the index build is the slow part)"
  if [ "$REBUILD" -eq 1 ] || [ ! -f "$FLAT" ]; then
    echo "   → ingest"
    python -m src.data.ingest --scope configs/phase1_scope.yaml
  else ok "flat.jsonl exists (skip ingest)"; fi

  if [ "$REBUILD" -eq 1 ] || [ ! -f "$NORM" ]; then
    echo "   → normalize"
    python -m src.data.normalize
  else ok "normalized.jsonl exists (skip normalize)"; fi

  if [ "$REBUILD" -eq 1 ] || [ ! -f "$READY" ]; then
    echo "   → build_index_text   (the step the README forgets)"
    python -m src.data.build_index_text
  else ok "ready_phase1.jsonl exists (skip build_index_text)"; fi

  if [ "$REBUILD" -eq 1 ] || [ ! -d "$CHROMA" ]; then
    echo "   → indexing.build     (downloads ~1.2 GB of BGE models on first run)"
    python -m src.indexing.build
  else ok "chroma_db_phase1/ exists (skip index build)"; fi
  ok "data pipeline complete"
fi

[ "$CHECK_ONLY" -eq 1 ] && { bold "preflight OK"; exit 0; }

# ----------------------------------------------------------------- 5. serve
[ "$SERVE" -eq 0 ] && { bold "done (--no-serve)"; exit 0; }

bold "5. Starting the API on http://127.0.0.1:8000"
echo "   first request waits ~30-60s while BGE-M3 + the reranker load"
echo "   docs:   http://127.0.0.1:8000/docs"
echo "   verify: curl localhost:8000/health && curl localhost:8000/taxonomy"
echo
exec python -m src.api --host 127.0.0.1 --port 8000
