"""Stage 3 — build the Chroma vector store from `data/processed/ready.jsonl`.

Run:
    python -m src.indexing.build

By default reads `configs/models.yaml`, embeds all rows with BGE-M3,
and writes the Chroma store to `data/vector_store/chroma_db/`.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any


def load_ready_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Ready JSONL not found: {path}")
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _run_smoke_test(
    *,
    model,
    vector_store,
    queries: list[dict],
    top_k: int,
) -> None:
    print("\n── Smoke test ──────────────────────────────")
    for q in queries:
        text = q["text"]
        lang = q.get("lang", "")
        vec = model.encode_query(text)
        where: dict[str, Any] | None = {"language": lang} if lang else None
        results = vector_store.collection.query(
            query_embeddings=[vec.tolist()],
            n_results=top_k,
            where=where,
            include=["metadatas", "documents", "distances"],
        )
        ids = (results.get("ids") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]
        docs = (results.get("documents") or [[]])[0]
        print(f"\n  [{lang}] query: {text!r}")
        for i, (_id, dist, doc) in enumerate(zip(ids, dists, docs), start=1):
            print(f"    {i}. id={_id[:28]:28s} dist={dist:+.3f} | {doc[:100]}")


def build(
    *,
    config_path: Path,
    input_path: Path,
    summary_path: Path,
) -> "object":
    # Lazy Pydantic + shared imports.
    from src.indexing.config import load_models_config
    from src.indexing.embedding_model import EmbeddingModel, EmbeddingModelConfig
    from src.indexing.taxonomy import Taxonomy
    from src.indexing.vector_store import (
        VectorStore,
        VectorStoreConfig,
        build_ids,
        row_to_metadata,
    )
    from src.shared.schemas import BuildVectorStoreStats, TaxonomyRecord

    config = load_models_config(config_path)
    print(f"Model           : {config.model.name}  (dim={config.model.embedding_dim})")
    print(f"Collection      : {config.vector_store.collection_name}")
    print(f"Persist dir     : {config.vector_store.persist_directory}")
    print(f"Reset on build  : {config.vector_store.reset_on_build}")
    print(f"Input rows from : {input_path}")

    rows = load_ready_rows(input_path)
    if not rows:
        raise ValueError(f"No rows in {input_path}")
    print(f"\nLoaded {len(rows)} rows.")

    taxonomy = Taxonomy.from_rows(rows)
    print(
        f"Taxonomy discovered: "
        f"{len(taxonomy.languages)} langs, "
        f"{len(taxonomy.subjects)} subjects, "
        f"{len(taxonomy.levels)} levels, "
        f"{len(taxonomy.question_types)} question types"
    )

    embedding_config = EmbeddingModelConfig(
        name=config.model.name,
        embedding_dim=config.model.embedding_dim,
        batch_size=config.model.batch_size,
        device=config.model.device,
        normalize_embeddings=config.model.normalize_embeddings,
        passage_prefix=config.model.passage_prefix,
        query_prefix=config.model.query_prefix,
    )
    print(f"\nLoading model on device={embedding_config.device}...")
    t_load_start = time.time()
    model = EmbeddingModel(embedding_config)
    print(f"  model ready in {time.time() - t_load_start:.1f}s (actual dim={model.dimension})")

    store_config = VectorStoreConfig(
        persist_directory=config.vector_store.persist_directory,
        collection_name=config.vector_store.collection_name,
        distance_metric=config.vector_store.distance_metric,
        add_batch_size=config.vector_store.add_batch_size,
        reset_on_build=config.vector_store.reset_on_build,
    )
    store = VectorStore(store_config)
    store.open()

    text_column = config.indexing.embedding_text_column
    scalar_fields = list(config.indexing.metadata_scalar_fields)
    list_fields_json = list(config.indexing.metadata_list_fields_as_json)
    list_fields_bools = list(config.indexing.metadata_list_fields_as_booleans)

    ids = build_ids(rows)
    by_language: Counter[str] = Counter()
    by_type: Counter[str] = Counter()
    rows_dropped: Counter[str] = Counter()
    total = len(rows)

    print(f"\nEmbedding {total} rows in batches of {store_config.add_batch_size}...")
    t_embed_start = time.time()

    # tqdm gives a single self-updating progress line in any environment
    # (terminal, IDE, notebook, redirected output — handles each gracefully).
    try:
        from tqdm import tqdm
        progress = tqdm(total=total, desc="Embedding", unit="rows", ncols=90)
    except ImportError:
        progress = None  # graceful fallback if tqdm unavailable

    for start in range(0, total, store_config.add_batch_size):
        end = min(start + store_config.add_batch_size, total)
        chunk = rows[start:end]
        chunk_ids = ids[start:end]

        chunk_texts = [str(r.get(text_column) or "").strip() for r in chunk]
        keep_indices = [i for i, t in enumerate(chunk_texts) if t]
        dropped_in_chunk = len(chunk) - len(keep_indices)
        if dropped_in_chunk:
            rows_dropped["empty_search_text"] += dropped_in_chunk

        if keep_indices:
            kept_texts = [chunk_texts[i] for i in keep_indices]
            kept_ids = [chunk_ids[i] for i in keep_indices]
            kept_rows = [chunk[i] for i in keep_indices]
            kept_metadata = [
                row_to_metadata(
                    r,
                    scalar_fields=scalar_fields,
                    list_fields_as_json=list_fields_json,
                    list_fields_as_booleans=list_fields_bools,
                )
                for r in kept_rows
            ]
            vectors = model.encode_passages(kept_texts, show_progress_bar=False)
            store.add_batch(
                ids=kept_ids,
                documents=kept_texts,
                metadatas=kept_metadata,
                embeddings=vectors.tolist(),
            )
            for r in kept_rows:
                by_language[str(r.get("language") or "unknown")] += 1
                by_type[str(r.get("question_type") or "unknown")] += 1

        if progress is not None:
            progress.update(end - start)

    if progress is not None:
        progress.close()

    wall = time.time() - t_embed_start
    rows_indexed = sum(by_language.values())
    rows_per_sec = round(rows_indexed / wall, 2) if wall > 0 else 0.0

    stats = BuildVectorStoreStats(
        rows_indexed=rows_indexed,
        model_name=config.model.name,
        embedding_dim=model.dimension,
        collection_name=config.vector_store.collection_name,
        persist_directory=str(config.vector_store.persist_directory),
        distance_metric=config.vector_store.distance_metric,
        wall_clock_seconds=round(wall, 1),
        rows_per_second=rows_per_sec,
        by_language=dict(by_language),
        by_question_type=dict(by_type),
        rows_dropped=dict(rows_dropped),
        taxonomy=TaxonomyRecord(**taxonomy.to_dict()),
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(stats.model_dump_json(indent=2), encoding="utf-8")

    print(f"\nIndexed {rows_indexed} rows in {wall:.1f}s ({rows_per_sec} rows/s).")
    print(f"By language: {dict(by_language)}")
    print(f"By type    : {dict(by_type)}")
    if rows_dropped:
        print(f"Dropped    : {dict(rows_dropped)}")
    print(f"Summary    : {summary_path}")

    if config.indexing.smoke_test.enabled and config.indexing.smoke_test.queries:
        _run_smoke_test(
            model=model,
            vector_store=store,
            queries=[q.model_dump() for q in config.indexing.smoke_test.queries],
            top_k=config.indexing.smoke_test.top_k,
        )

    return stats


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/models.yaml"))
    p.add_argument("--input", type=Path, default=Path("data/processed/ready.jsonl"))
    p.add_argument(
        "--summary",
        type=Path,
        default=Path("data/vector_store/build_summary.json"),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    build(
        config_path=args.config,
        input_path=args.input,
        summary_path=args.summary,
    )


if __name__ == "__main__":
    main()
