"""Stage 4 — Retriever.

Wraps Stage 3's vector store into a typed, filtered, deduplicated retrieval
API suitable for the Stage 5 LLM generator. Rewritten from scratch on top
of the single-model BGE-M3 architecture (no more profile routing).

Responsibilities:
  - Load payload (ready.jsonl) into memory, keyed by doc_id
  - Load taxonomy from build_summary.json for input validation
  - Normalize queries (LaTeX) for symmetry with the corpus
  - Build Chroma `where` clauses from scalar + boolean-per-level filters
  - Post-query: script-mismatch guard, quiz-title dedup, max-distance cutoff
  - Join payload to return typed RetrievedQuestion objects
  - Expose taxonomy listings for frontend dropdowns

NOT responsible for (kept elsewhere by design):
  - Filter relaxation fallback (deferred; may add in Stage 7)
  - Reranking / hybrid sparse retrieval (out of scope)
  - Query expansion / translation (out of scope)
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any

from src.data.latex import normalize_latex
from src.retrieval.schemas import RetrievedQuestion


ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
LATIN_CHAR_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]")


def _detect_dominant_script(text: str) -> str:
    """Return 'arabic' | 'latin' | 'mixed' | 'none' based on character counts."""
    arabic_count = len(ARABIC_CHAR_RE.findall(text))
    latin_count = len(LATIN_CHAR_RE.findall(text))
    if arabic_count == 0 and latin_count == 0:
        return "none"
    if arabic_count >= 3 and arabic_count > latin_count * 1.2:
        return "arabic"
    if latin_count >= 3 and latin_count > arabic_count * 1.2:
        return "latin"
    return "mixed"


def _row_matches_requested_language(
    *,
    language: str,
    question_text: str,
    choices: list[str],
    correct_answers: list[str],
) -> bool:
    """Guard against mislabeled rows whose content is in a different script."""
    combined = " ".join([question_text, *choices, *correct_answers]).strip()
    script = _detect_dominant_script(combined)
    normalized = language.strip().lower()
    if normalized in {"en", "fr"} and script == "arabic":
        return False
    if normalized == "ar" and script == "latin":
        return False
    return True


def _build_where(
    *,
    language: str | None,
    question_type: str | None,
    multiple_correct_answers: bool | None,
    subject: str | None,
    levels: list[str] | None,
    levels_match_mode: str,
) -> dict[str, Any] | None:
    """Compose a Chroma `where` dict from scalar + boolean-per-level filters."""
    clauses: list[dict[str, Any]] = []
    if language:
        clauses.append({"language": language})
    if question_type:
        clauses.append({"question_type": question_type})
    if multiple_correct_answers is not None:
        clauses.append({"multiple_correct_answers": multiple_correct_answers})
    if subject:
        clauses.append({"subject": subject})
    if levels:
        level_clauses = [{f"levels_{lvl}": True} for lvl in levels if lvl]
        if len(level_clauses) == 1:
            clauses.append(level_clauses[0])
        elif len(level_clauses) > 1:
            operator = "$and" if levels_match_mode == "all" else "$or"
            clauses.append({operator: level_clauses})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _row_to_retrieved(row: dict, distance: float) -> RetrievedQuestion:
    """Join a payload row with a Chroma distance into a typed RetrievedQuestion."""
    return RetrievedQuestion(
        doc_id=str(row.get("doc_id", "")),
        quiz_id=str(row.get("quiz_id", "")),
        quiz_title=str(row.get("quiz_title", "")),
        language=str(row.get("language", "")),
        question_type=str(row.get("question_type", "")),
        question_text=str(row.get("question_text", "")),
        choices=list(row.get("choices_text") or []),
        correct_answers=list(row.get("correct_choices_text") or []),
        subjects=list(row.get("subjects") or []),
        levels=list(row.get("levels") or []),
        multiple_correct_answers=bool(row.get("multiple_correct_answers", False)),
        author_name=row.get("author_name"),
        author_email=row.get("author_email"),
        search_text=str(row.get("search_text", "")),
        metadata=dict(row),
        distance=float(distance),
    )


class Retriever:
    """Load-once, query-many retrieval over the Chroma vector store."""

    def __init__(
        self,
        config_path: Path | str = Path("configs/models.yaml"),
        ready_jsonl_path: Path | str = Path("data/processed/ready.jsonl"),
        *,
        _model: Any | None = None,
        _collection: Any | None = None,
        _taxonomy: Any | None = None,
        _payload: dict[str, dict] | None = None,
        _reranker: Any | None = None,
    ) -> None:
        """Initialize the retriever.

        Under normal use, pass only config_path and ready_jsonl_path.
        The underscored kwargs exist to support unit tests that inject mocks
        instead of loading the model/collection/payload from disk.
        """
        self.config_path = Path(config_path)
        self.ready_jsonl_path = Path(ready_jsonl_path)
        # Pool size used when reranker is enabled — Chroma fetches more
        # candidates so the reranker has a wider set to score and reorder.
        self._reranker_candidate_pool: int = 50

        if _model is not None and _collection is not None:
            # Test injection path — skip all disk loading.
            self._model = _model
            self._collection = _collection
            self._taxonomy = _taxonomy
            self._payload = _payload or {}
            self._reranker = _reranker
            return

        # Production path — check prerequisites then load.
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Models config not found: {self.config_path}. "
                "Have you created configs/models.yaml?"
            )
        if not self.ready_jsonl_path.exists():
            raise FileNotFoundError(
                f"Payload JSONL not found: {self.ready_jsonl_path}. "
                "Run `python -m src.data.build_index_text` first."
            )

        from src.indexing.config import load_models_config
        from src.indexing.embedding_model import EmbeddingModel, EmbeddingModelConfig
        from src.indexing.taxonomy import Taxonomy
        from src.indexing.vector_store import VectorStore, VectorStoreConfig

        self.config = load_models_config(self.config_path)

        persist_dir = self.config.vector_store.persist_directory
        if not persist_dir.exists():
            raise FileNotFoundError(
                f"Vector store not found: {persist_dir}. "
                "Run `python -m src.indexing.build` first."
            )

        ec = EmbeddingModelConfig(
            name=self.config.model.name,
            embedding_dim=self.config.model.embedding_dim,
            batch_size=self.config.model.batch_size,
            device=self.config.model.device,
            normalize_embeddings=self.config.model.normalize_embeddings,
            passage_prefix=self.config.model.passage_prefix,
            query_prefix=self.config.model.query_prefix,
        )
        self._model = EmbeddingModel(ec)

        sc = VectorStoreConfig(
            persist_directory=self.config.vector_store.persist_directory,
            collection_name=self.config.vector_store.collection_name,
            distance_metric=self.config.vector_store.distance_metric,
            add_batch_size=self.config.vector_store.add_batch_size,
            reset_on_build=False,   # never reset at query time
        )
        store = VectorStore(sc)
        store.open()
        self._collection = store.collection

        summary_path = persist_dir.parent / "build_summary.json"
        self._taxonomy = Taxonomy.from_build_summary(summary_path)

        # Payload — load all rows into memory, dict by doc_id (~50 MB at 10k rows)
        self._payload = self._load_payload(self.ready_jsonl_path)

        # Optional reranker — loaded only when configured `enabled: true`.
        self._reranker = None
        rr_cfg = getattr(self.config, "reranker", None)
        if rr_cfg is not None and getattr(rr_cfg, "enabled", False):
            from src.retrieval.reranker import Reranker, RerankerConfig
            self._reranker = Reranker(
                RerankerConfig(
                    model_name=rr_cfg.model_name,
                    device=rr_cfg.device,
                    batch_size=rr_cfg.batch_size,
                )
            )
            self._reranker_candidate_pool = rr_cfg.candidate_pool_size

    @staticmethod
    def _load_payload(path: Path) -> dict[str, dict]:
        payload: dict[str, dict] = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                doc_id = row.get("doc_id")
                if doc_id:
                    payload[str(doc_id)] = row
        return payload

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_languages(self) -> list[str]:
        return self._taxonomy.list_languages() if self._taxonomy else []

    def list_question_types(self) -> list[str]:
        return self._taxonomy.list_question_types() if self._taxonomy else []

    def list_subjects(self) -> list[str]:
        return self._taxonomy.list_subjects() if self._taxonomy else []

    def list_levels(self) -> list[str]:
        return self._taxonomy.list_levels() if self._taxonomy else []

    # ------------------------------------------------------------------
    # Diagnostics (why did retrieval return empty?)
    # ------------------------------------------------------------------

    def _count_matching(self, where: dict[str, Any] | None) -> int | None:
        """Count rows matching a Chroma where clause. Returns None on error."""
        if self._collection is None:
            return None
        try:
            if where is None:
                return self._collection.count()
            result = self._collection.get(where=where, include=[])
            ids = result.get("ids", []) if isinstance(result, dict) else []
            return len(ids)
        except Exception:
            return None

    def diagnose_empty(
        self,
        *,
        language: str | None = None,
        question_type: str | None = None,
        subject: str | None = None,
        levels: list[str] | None = None,
    ) -> str:
        """Explain why retrieval returned empty in human terms.

        Shows: total rows, single-filter counts, and — when a subject is
        given — how many rows are available in each language for that subject
        (so the caller can pick a working combo).
        """
        lines = ["Diagnostic — why no rows matched:"]
        total = self._count_matching(None)
        if total is not None:
            lines.append(f"  Total rows in corpus: {total:,}")

        # Single-filter counts
        single_filters: list[tuple[str, dict[str, Any]]] = []
        if language:
            single_filters.append((f"language={language!r}", {"language": language}))
        if question_type:
            single_filters.append((f"question_type={question_type!r}", {"question_type": question_type}))
        if subject:
            subject_upper = subject.strip().upper()
            single_filters.append((f"subject={subject_upper!r}", {"subject": subject_upper}))
        if levels:
            for lvl in levels:
                single_filters.append((f"level={lvl!r}", {f"levels_{lvl}": True}))

        if single_filters:
            lines.append("  Rows matching each filter alone:")
            for label, where in single_filters:
                n = self._count_matching(where)
                lines.append(f"    - {label}: {n if n is not None else '?'}")

        # Language cross-tab for the requested subject — suggests alternatives
        if subject:
            subject_upper = subject.strip().upper()
            lines.append(f"  Languages available for subject={subject_upper!r}:")
            for lang in ("en", "fr", "ar"):
                n = self._count_matching(
                    {"$and": [{"subject": subject_upper}, {"language": lang}]}
                )
                arrow = " ← requested" if language == lang else ""
                lines.append(f"    - {lang}: {n if n is not None else '?'}{arrow}")

        # Suggest alternatives
        suggestions: list[str] = []
        if subject and language:
            subject_upper = subject.strip().upper()
            for lang in ("fr", "ar", "en"):
                if lang == language:
                    continue
                n = self._count_matching(
                    {"$and": [{"subject": subject_upper}, {"language": lang}]}
                )
                if n and n > 0:
                    suggestions.append(f"language={lang!r} ({n:,} rows)")
        if suggestions:
            lines.append(
                f"  Try one of these instead: {', '.join(suggestions)}"
            )

        return "\n".join(lines)

    def retrieve(
        self,
        query: str,
        *,
        language: str,
        top_k: int = 5,
        candidate_pool_size: int = 50,
        max_distance: float | None = None,
        question_type: str | None = None,
        multiple_correct_answers: bool | None = None,
        subject: str | None = None,
        levels: list[str] | None = None,
        levels_match_mode: str = "any",
        author_name: str | None = None,
        quiz_title_contains: str | None = None,
        dedup_by_quiz_title: bool = True,
    ) -> list[RetrievedQuestion]:
        """Return up to top_k matching RetrievedQuestion rows."""
        if not query or not query.strip():
            raise ValueError("query is required and cannot be empty")
        if not language or not language.strip():
            raise ValueError("language is required (one of en/fr/ar)")
        if levels_match_mode not in {"any", "all"}:
            raise ValueError(
                f"levels_match_mode must be 'any' or 'all', got {levels_match_mode!r}"
            )

        # Validate inputs against the known taxonomy (warnings only).
        if self._taxonomy is not None:
            self._taxonomy.validate_language(language)
            self._taxonomy.validate_question_type(question_type)
            if subject:
                self._taxonomy.validate_subject(subject.strip().upper())
            self._taxonomy.validate_levels(levels or [])

        # Normalize LaTeX in the query (symmetry with the corpus).
        normalized_query = normalize_latex(query)

        where = _build_where(
            language=language,
            question_type=question_type,
            multiple_correct_answers=multiple_correct_answers,
            subject=subject.strip().upper() if subject else None,
            levels=list(levels or []),
            levels_match_mode=levels_match_mode,
        )

        q_vec = self._model.encode_query(normalized_query)
        # If a reranker is loaded, fetch a wider candidate pool so the
        # reranker has more to score; otherwise use the caller's pool size.
        if self._reranker is not None:
            pool_size = max(self._reranker_candidate_pool, candidate_pool_size, top_k)
        else:
            pool_size = max(candidate_pool_size, top_k)

        collection_count = self._collection.count()
        if collection_count == 0:
            warnings.warn(
                "Vector store is empty. Build it with `python -m src.indexing.build`."
            )
            return []
        n_results = min(pool_size, collection_count)

        q_embedding = q_vec.tolist() if hasattr(q_vec, "tolist") else list(q_vec)
        results = self._collection.query(
            query_embeddings=[q_embedding],
            n_results=n_results,
            where=where,
            include=["metadatas", "documents", "distances"],
        )

        ids = (results.get("ids") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        output: list[RetrievedQuestion] = []
        seen_dedup_keys: set[str] = set()
        # When the reranker is on, collect ALL surviving candidates (so the
        # reranker can reorder a wider pool); otherwise stop early at top_k
        # for a small performance win.
        target_size = pool_size if self._reranker is not None else top_k

        for doc_id, distance in zip(ids, distances):
            # Distance cutoff — Chroma returns sorted ascending, so break early.
            if max_distance is not None and distance > max_distance:
                break

            row = self._payload.get(doc_id)
            if row is None:
                # Index drift — row in Chroma but missing from payload.
                continue

            if quiz_title_contains:
                quiz_title = (row.get("quiz_title") or "").casefold()
                if quiz_title_contains.casefold() not in quiz_title:
                    continue

            if author_name:
                row_author = (row.get("author_name") or "").strip().casefold()
                if row_author != author_name.strip().casefold():
                    continue

            if not _row_matches_requested_language(
                language=language,
                question_text=str(row.get("question_text") or ""),
                choices=list(row.get("choices_text") or []),
                correct_answers=list(row.get("correct_choices_text") or []),
            ):
                continue

            if dedup_by_quiz_title:
                qt = (row.get("quiz_title") or "").casefold().strip()
                qtext = (row.get("question_text") or "").casefold().strip()
                dedup_key = f"{qt}|{qtext}"
                if dedup_key in seen_dedup_keys:
                    continue
                seen_dedup_keys.add(dedup_key)

            output.append(_row_to_retrieved(row, distance))
            if len(output) >= target_size:
                break

        # Cross-encoder reranking step (only if a reranker is configured).
        # Filters have already been applied — reranker only reorders.
        if self._reranker is not None and len(output) > 1:
            output = self._reranker.rerank(query, output)
        # Slice to top_k after reranking (or after early break in no-reranker path).
        output = output[:top_k]

        if not output:
            diagnostic = self.diagnose_empty(
                language=language,
                question_type=question_type,
                subject=subject,
                levels=levels,
            )
            warnings.warn(
                f"No retrieval results for query={query!r} with filters="
                f"{{'language': {language!r}, 'subject': {subject!r}, "
                f"'levels': {levels!r}, 'question_type': {question_type!r}}}"
                f"\n{diagnostic}"
            )

        return output

    def batch_retrieve(
        self,
        queries: list[dict],
    ) -> list[list[RetrievedQuestion]]:
        """Run `retrieve()` for each item in `queries`.

        Each item is a dict with at least 'query' and 'language' keys;
        other keys pass through as kwargs.
        """
        results: list[list[RetrievedQuestion]] = []
        for q in queries:
            if not isinstance(q, dict):
                raise ValueError("batch_retrieve items must be dicts")
            q_copy = dict(q)
            text = q_copy.pop("query", None)
            if text is None:
                raise ValueError("batch_retrieve item missing 'query' key")
            results.append(self.retrieve(text, **q_copy))
        return results
