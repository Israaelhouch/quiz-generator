"""Tests for Stage 4 — Retriever.

Uses mock EmbeddingModel and mock Chroma collection via the Retriever's
underscored test-injection kwargs. No real model or Chroma is loaded.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.indexing.taxonomy import Taxonomy
from src.retrieval.retriever import (
    Retriever,
    _build_where,
    _detect_dominant_script,
    _row_matches_requested_language,
)
from src.retrieval.schemas import RetrievedQuestion


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


class FakeModel:
    """Stand-in for EmbeddingModel. Returns a deterministic vector."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def encode_query(self, text: str):
        self.calls.append(text)
        return [0.1, 0.2, 0.3]


class FakeCollection:
    """Stand-in for Chroma collection."""

    def __init__(self, ids: list[str], distances: list[float]) -> None:
        self._ids = ids
        self._distances = distances
        self.calls: list[dict] = []

    def count(self) -> int:
        return len(self._ids)

    def query(self, *, query_embeddings, n_results, where, include):
        self.calls.append(
            {
                "query_embeddings": query_embeddings,
                "n_results": n_results,
                "where": where,
                "include": include,
            }
        )
        cap = min(n_results, len(self._ids))
        return {
            "ids": [self._ids[:cap]],
            "distances": [self._distances[:cap]],
            "metadatas": [[{} for _ in range(cap)]],
            "documents": [["" for _ in range(cap)]],
        }


def _payload_row(
    doc_id: str,
    *,
    quiz_title: str = "Quiz A",
    question_text: str = "What is X?",
    language: str = "en",
    question_type: str = "MULTIPLE_CHOICE",
    choices: list[str] | None = None,
    correct: list[str] | None = None,
    subjects: list[str] | None = None,
    levels: list[str] | None = None,
    multiple_correct_answers: bool = False,
    author_name: str | None = "Prof",
) -> dict:
    return {
        "doc_id": doc_id,
        "quiz_id": f"quiz-{doc_id}",
        "quiz_title": quiz_title,
        "language": language,
        "question_type": question_type,
        "question_text": question_text,
        "choices_text": choices or ["A", "B", "C"],
        "correct_choices_text": correct or ["A"],
        "subjects": subjects or ["SCIENCE"],
        "levels": levels or ["PRIMARY_SCHOOL_6TH_GRADE"],
        "multiple_correct_answers": multiple_correct_answers,
        "author_name": author_name,
        "author_email": "x@y.z",
        "search_text": f"{quiz_title}. {question_text}",
    }


def _make_retriever(
    *,
    ids: list[str],
    distances: list[float],
    payload: dict[str, dict],
    taxonomy: Taxonomy | None = None,
) -> Retriever:
    """Build a Retriever with injected fakes — skips disk/model loading."""
    return Retriever(
        config_path=Path("unused"),
        ready_jsonl_path=Path("unused"),
        _model=FakeModel(),
        _collection=FakeCollection(ids=ids, distances=distances),
        _taxonomy=taxonomy or Taxonomy(
            languages={"en", "fr", "ar"},
            question_types={"MULTIPLE_CHOICE", "FILL_IN_THE_BLANKS"},
            subjects={"SCIENCE", "MATHEMATICS", "PHYSICS"},
            levels={"PRIMARY_SCHOOL_6TH_GRADE", "HIGH_SCHOOL_4TH_GRADE_MATHEMATICS"},
        ),
        _payload=payload,
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def test_detect_dominant_script() -> None:
    assert _detect_dominant_script("Hello world") == "latin"
    assert _detect_dominant_script("مرحبا بك في العربية") == "arabic"
    assert _detect_dominant_script("") == "none"
    # Mixed → not either
    assert _detect_dominant_script("Hello مرحبا") == "mixed"


def test_row_matches_requested_language() -> None:
    # English label, English content → match
    assert _row_matches_requested_language(
        language="en", question_text="What is X?", choices=["A", "B"], correct_answers=["A"]
    )
    # English label, Arabic content → reject
    assert not _row_matches_requested_language(
        language="en",
        question_text="ما هو الجهاز المناعي؟",
        choices=["أ", "ب"],
        correct_answers=["أ"],
    )
    # Arabic label, Latin content → reject
    assert not _row_matches_requested_language(
        language="ar",
        question_text="What is X?",
        choices=["A", "B"],
        correct_answers=["A"],
    )


def test_build_where_scalar_only() -> None:
    where = _build_where(
        language="fr",
        question_type="MULTIPLE_CHOICE",
        multiple_correct_answers=None,
        subject=None,
        levels=None,
        levels_match_mode="any",
    )
    assert where == {"$and": [{"language": "fr"}, {"question_type": "MULTIPLE_CHOICE"}]}


def test_build_where_single_scalar_returns_flat() -> None:
    where = _build_where(
        language="fr", question_type=None, multiple_correct_answers=None,
        subject=None, levels=None, levels_match_mode="any",
    )
    assert where == {"language": "fr"}


def test_build_where_levels_any_mode() -> None:
    where = _build_where(
        language="fr", question_type=None, multiple_correct_answers=None, subject=None,
        levels=["L1", "L2"], levels_match_mode="any",
    )
    # fr + ($or of two level booleans) combined with $and
    assert where == {
        "$and": [
            {"language": "fr"},
            {"$or": [{"levels_L1": True}, {"levels_L2": True}]},
        ]
    }


def test_build_where_levels_all_mode() -> None:
    where = _build_where(
        language=None, question_type=None, multiple_correct_answers=None, subject=None,
        levels=["L1", "L2"], levels_match_mode="all",
    )
    assert where == {"$and": [{"levels_L1": True}, {"levels_L2": True}]}


def test_build_where_no_filters_returns_none() -> None:
    where = _build_where(
        language=None, question_type=None, multiple_correct_answers=None, subject=None,
        levels=None, levels_match_mode="any",
    )
    assert where is None


# ---------------------------------------------------------------------------
# Retriever.retrieve — behavior
# ---------------------------------------------------------------------------


def test_retrieve_rejects_empty_query() -> None:
    r = _make_retriever(ids=[], distances=[], payload={})
    try:
        r.retrieve("", language="en")
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "query" in str(exc).lower()


def test_retrieve_rejects_empty_language() -> None:
    r = _make_retriever(ids=[], distances=[], payload={})
    try:
        r.retrieve("x", language="")
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "language" in str(exc).lower()


def test_retrieve_empty_store_warns_and_returns_empty() -> None:
    r = _make_retriever(ids=[], distances=[], payload={})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = r.retrieve("anything", language="en")
    assert result == []
    assert any("empty" in str(w.message).lower() for w in caught)


def test_retrieve_joins_payload_into_typed_result() -> None:
    payload = {
        "id-1": _payload_row("id-1", question_text="Q1", choices=["A", "B"], correct=["A"]),
    }
    r = _make_retriever(ids=["id-1"], distances=[0.25], payload=payload)
    results = r.retrieve("x", language="en", top_k=1)
    assert len(results) == 1
    rq = results[0]
    assert isinstance(rq, RetrievedQuestion)
    assert rq.doc_id == "id-1"
    assert rq.choices == ["A", "B"]
    assert rq.correct_answers == ["A"]
    assert rq.distance == 0.25
    assert rq.author_name == "Prof"


def test_retrieve_applies_max_distance() -> None:
    payload = {
        "id-1": _payload_row("id-1"),
        "id-2": _payload_row("id-2", question_text="Q2"),
    }
    r = _make_retriever(ids=["id-1", "id-2"], distances=[0.3, 0.9], payload=payload)
    results = r.retrieve("x", language="en", top_k=5, max_distance=0.5)
    assert len(results) == 1
    assert results[0].doc_id == "id-1"


def test_retrieve_dedups_same_quiz_title_and_question() -> None:
    payload = {
        "id-1": _payload_row("id-1", quiz_title="Immunity 1", question_text="What is a pathogen?"),
        "id-2": _payload_row("id-2", quiz_title="Immunity 1", question_text="What is a pathogen?"),
        "id-3": _payload_row("id-3", quiz_title="Immunity 2", question_text="What is an antigen?"),
    }
    r = _make_retriever(ids=["id-1", "id-2", "id-3"], distances=[0.1, 0.2, 0.3], payload=payload)
    results = r.retrieve("immune", language="en", top_k=5)
    assert len(results) == 2  # id-1 survives, id-2 is dedup'd, id-3 distinct
    assert results[0].doc_id == "id-1"
    assert results[1].doc_id == "id-3"


def test_retrieve_script_mismatch_guard_drops_wrong_language_rows() -> None:
    payload = {
        "id-1": _payload_row(
            "id-1",
            language="en",
            question_text="ما هي وظيفة المناعة؟",  # Arabic content
            choices=["أ", "ب"],
            correct=["أ"],
        ),
        "id-2": _payload_row("id-2", question_text="What is immunity?"),
    }
    r = _make_retriever(ids=["id-1", "id-2"], distances=[0.2, 0.3], payload=payload)
    results = r.retrieve("x", language="en", top_k=5)
    assert len(results) == 1
    assert results[0].doc_id == "id-2"


def test_retrieve_quiz_title_contains_filter() -> None:
    payload = {
        "id-1": _payload_row("id-1", quiz_title="Math Basics"),
        "id-2": _payload_row("id-2", quiz_title="Physics Advanced"),
    }
    r = _make_retriever(ids=["id-1", "id-2"], distances=[0.1, 0.2], payload=payload)
    results = r.retrieve("anything", language="en", quiz_title_contains="math")
    assert len(results) == 1
    assert "Math" in results[0].quiz_title


def test_retrieve_author_name_filter() -> None:
    payload = {
        "id-1": _payload_row("id-1", author_name="Alice"),
        "id-2": _payload_row("id-2", author_name="Bob"),
    }
    r = _make_retriever(ids=["id-1", "id-2"], distances=[0.1, 0.2], payload=payload)
    results = r.retrieve("q", language="en", author_name="Alice")
    assert len(results) == 1
    assert results[0].author_name == "Alice"


def test_retrieve_builds_where_clause_with_subject_and_levels() -> None:
    payload = {"id-1": _payload_row("id-1")}
    r = _make_retriever(ids=["id-1"], distances=[0.1], payload=payload)
    r.retrieve(
        "q", language="fr", subject="MATHEMATICS",
        levels=["HIGH_SCHOOL_4TH_GRADE_MATHEMATICS"],
    )
    call = r._collection.calls[0]
    assert call["where"] == {
        "$and": [
            {"language": "fr"},
            {"subject": "MATHEMATICS"},
            {"levels_HIGH_SCHOOL_4TH_GRADE_MATHEMATICS": True},
        ]
    }


def test_retrieve_normalizes_latex_in_query() -> None:
    """Query with LaTeX markup gets normalized before encoding (symmetry)."""
    payload = {"id-1": _payload_row("id-1")}
    r = _make_retriever(ids=["id-1"], distances=[0.1], payload=payload)
    r.retrieve(r"Calculer \(\sin(x)\)", language="fr")
    # FakeModel.calls captured the text passed to encode_query.
    encoded = r._model.calls[0]
    assert "\\sin" not in encoded  # LaTeX stripped
    assert "sin" in encoded  # function name preserved


def test_retrieve_index_drift_skips_missing_payload() -> None:
    # Chroma returns id-missing but payload has no such doc_id
    r = _make_retriever(
        ids=["id-missing", "id-ok"],
        distances=[0.1, 0.2],
        payload={"id-ok": _payload_row("id-ok")},
    )
    results = r.retrieve("q", language="en")
    assert len(results) == 1
    assert results[0].doc_id == "id-ok"


# ---------------------------------------------------------------------------
# Reranker integration
# ---------------------------------------------------------------------------


class _FakeReranker:
    """Stand-in for the Reranker class — reorders by canned scores per text."""

    def __init__(self, scores_by_text: dict[str, float]) -> None:
        self.scores_by_text = scores_by_text
        self.calls: list[dict] = []

    def rerank(self, query: str, candidates):
        self.calls.append({"query": query, "n": len(candidates)})
        scored = [(c, self.scores_by_text.get(c.search_text, 0.0)) for c in candidates]
        scored.sort(key=lambda x: -x[1])
        return [c for c, _ in scored]


def test_retrieve_reranker_reorders_results() -> None:
    """When a reranker is injected, candidates are returned in score order
    (descending) — NOT in original Chroma distance order."""
    payload = {
        "id-a": _payload_row("id-a", quiz_title="QA", question_text="alpha"),
        "id-b": _payload_row("id-b", quiz_title="QB", question_text="beta"),
        "id-c": _payload_row("id-c", quiz_title="QC", question_text="gamma"),
    }
    # Distances ascending = a, b, c (Chroma's preferred order)
    r = _make_retriever(
        ids=["id-a", "id-b", "id-c"],
        distances=[0.1, 0.2, 0.3],
        payload=payload,
    )
    # Inject a reranker that gives c the highest score, then a, then b
    fake = _FakeReranker(
        {
            "QA. alpha": 0.3,
            "QB. beta": 0.1,
            "QC. gamma": 0.9,
        }
    )
    r._reranker = fake
    r._reranker_candidate_pool = 50

    results = r.retrieve("anything", language="en", top_k=3)
    # Reranker order: c (0.9) > a (0.3) > b (0.1)
    assert [x.doc_id for x in results] == ["id-c", "id-a", "id-b"]
    assert len(fake.calls) == 1
    assert fake.calls[0]["n"] == 3  # all 3 candidates passed to reranker


def test_retrieve_reranker_widens_candidate_pool() -> None:
    """Reranker enabled → Chroma should be queried with the wider pool size,
    not just top_k."""
    # Need a corpus larger than the pool size so n_results isn't capped
    payload = {f"id-{i}": _payload_row(f"id-{i}", question_text=f"q{i}") for i in range(100)}
    r = _make_retriever(
        ids=[f"id-{i}" for i in range(100)],
        distances=[0.001 * i for i in range(100)],
        payload=payload,
    )
    r._reranker = _FakeReranker({})  # canned scores: all 0 (stable order)
    r._reranker_candidate_pool = 50

    r.retrieve("q", language="en", top_k=3)
    # n_results passed to Chroma should be at least the pool size, not top_k
    assert r._collection.calls[0]["n_results"] >= 50


def test_retrieve_no_reranker_skips_rerank_call() -> None:
    """No reranker configured → no reranker call, results in distance order."""
    payload = {
        "id-a": _payload_row("id-a", question_text="alpha"),
        "id-b": _payload_row("id-b", question_text="beta"),
    }
    r = _make_retriever(
        ids=["id-a", "id-b"], distances=[0.1, 0.2], payload=payload
    )
    # _reranker stays None on the retriever (default)
    assert r._reranker is None
    results = r.retrieve("q", language="en", top_k=2)
    # Distance order preserved
    assert [x.doc_id for x in results] == ["id-a", "id-b"]


# ---------------------------------------------------------------------------
# batch_retrieve
# ---------------------------------------------------------------------------


def test_batch_retrieve_runs_all_queries() -> None:
    payload = {"id-1": _payload_row("id-1")}
    r = _make_retriever(ids=["id-1"], distances=[0.1], payload=payload)
    results = r.batch_retrieve(
        [
            {"query": "first", "language": "en", "top_k": 1},
            {"query": "second", "language": "en", "top_k": 1},
        ]
    )
    assert len(results) == 2
    assert len(r._model.calls) == 2


def test_batch_retrieve_rejects_missing_query_key() -> None:
    r = _make_retriever(ids=[], distances=[], payload={})
    try:
        r.batch_retrieve([{"language": "en"}])
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "query" in str(exc).lower()


# ---------------------------------------------------------------------------
# Taxonomy listing passthrough
# ---------------------------------------------------------------------------


def test_diagnose_empty_shows_filter_counts_and_suggestions() -> None:
    """The empty-result diagnostic shows single-filter counts and suggests
    alternative languages for the same subject.
    """

    class CountingCollection(FakeCollection):
        """Track what `get(where=...)` queries are made and return canned counts."""
        def __init__(self) -> None:
            super().__init__(ids=[], distances=[])
            self._count = 10000
            self.get_calls: list[dict] = []

        def get(self, *, where=None, include=None):
            self.get_calls.append({"where": where, "include": include})
            # Canned counts keyed by the filter shape
            canned: dict[str, int] = {
                "language=en": 4489,
                "language=fr": 3660,
                "subject=MATHEMATICS": 1672,
                "question_type=MULTIPLE_CHOICE": 10000,
                "MATHEMATICS+en": 0,      # the missing combo (subject-first in $and)
                "MATHEMATICS+fr": 1148,
                "MATHEMATICS+ar": 524,
            }
            key = _where_key(where)
            return {"ids": ["x"] * canned.get(key, 0), "documents": [], "metadatas": []}

    def _where_key(where):
        if not where:
            return ""
        if "$and" in where:
            fields = []
            for clause in where["$and"]:
                for k, v in clause.items():
                    if k == "language":
                        fields.append(v)
                    elif k == "subject":
                        fields.append(v)
            return "+".join(fields)
        for k, v in where.items():
            return f"{k}={v}"
        return ""

    r = Retriever.__new__(Retriever)
    r.config_path = Path("x")
    r.ready_jsonl_path = Path("x")
    r._model = FakeModel()
    r._collection = CountingCollection()
    r._taxonomy = None
    r._payload = {}

    msg = r.diagnose_empty(
        language="en",
        question_type="MULTIPLE_CHOICE",
        subject="MATHEMATICS",
    )
    assert "Total rows in corpus" in msg
    assert "language='en'" in msg
    assert "subject='MATHEMATICS'" in msg
    # Shows the language cross-tab for the subject
    assert "Languages available for subject='MATHEMATICS'" in msg
    # Suggests alternatives when the requested combo is empty
    assert "Try one of these instead" in msg
    assert "fr'" in msg  # fr is suggested since it has content for MATHEMATICS


def test_diagnose_empty_gracefully_handles_exceptions() -> None:
    """If the collection call fails, diagnose returns something reasonable."""

    class BrokenCollection:
        def count(self):
            raise RuntimeError("chroma down")
        def get(self, **_):
            raise RuntimeError("chroma down")

    r = Retriever.__new__(Retriever)
    r._model = FakeModel()
    r._collection = BrokenCollection()
    r._taxonomy = None
    r._payload = {}

    msg = r.diagnose_empty(language="en", subject="MATHEMATICS")
    # Shouldn't raise; string is returned even without real counts
    assert isinstance(msg, str)
    assert "Diagnostic" in msg


def test_list_methods_delegate_to_taxonomy() -> None:
    r = _make_retriever(ids=[], distances=[], payload={})
    assert r.list_languages() == ["ar", "en", "fr"]
    assert "MATHEMATICS" in r.list_subjects()
    assert "PRIMARY_SCHOOL_6TH_GRADE" in r.list_levels()


if __name__ == "__main__":
    import inspect
    mod = sys.modules[__name__]
    for name, fn in sorted(inspect.getmembers(mod, inspect.isfunction)):
        if name.startswith("test_"):
            fn()
    print("All Stage 4 (retriever) tests passed.")
