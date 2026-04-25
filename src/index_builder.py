#!/usr/bin/env python3
"""
index_builder.py
PDF -> markdown text -> chunks -> embeddings -> BM25 + FAISS + metadata

Entry point (called by main.py):
    build_index(markdown_file, cfg, keep_tables=True, do_visualize=False)
"""

import os
import pickle
import pathlib
import re
import json
from dataclasses import asdict
from typing import List, Dict

import faiss
from rank_bm25 import BM25Okapi
from src.embedder import SentenceTransformer

from src.preprocessing.chunking import DocumentChunker, ChunkConfig
from src.preprocessing.extraction import extract_sections_from_markdown
from src.preprocessing.docling_extraction_chunk import chunk_docling_hierarchical_json

# ----- runtime parallelism knobs (avoid oversubscription) -----
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Default keywords to exclude sections
DEFAULT_EXCLUSION_KEYWORDS = ['questions', 'exercises', 'summary', 'references']
DEFAULT_PGVECTOR_TABLE = "docling_pg_chunks"

# ------------------------ Main index builder -----------------------------

def build_index(
    markdown_file: str,
    *,
    chunker: DocumentChunker,
    chunk_config: ChunkConfig,
    embedding_model_path: str,
    artifacts_dir: os.PathLike,
    index_prefix: str,
    use_multiprocessing: bool = False,
    use_headings: bool = False,
) -> None:
    """
    Extract sections, chunk, embed, and build both FAISS and BM25 indexes.

    Persists:
        - {prefix}.faiss
        - {prefix}_bm25.pkl
        - {prefix}_chunks.pkl
        - {prefix}_sources.pkl
        - {prefix}_meta.pkl
    """
    all_chunks: List[str] = []
    sources: List[str] = []
    metadata: List[Dict] = []

    # Extract sections from markdown. Exclude some with certain keywords.
    sections = extract_sections_from_markdown(
        markdown_file,
        exclusion_keywords=DEFAULT_EXCLUSION_KEYWORDS
    )

    page_to_chunk_ids = {}
    current_page = 1
    total_chunks = 0
    heading_stack = []

    print("Using standard chunking strategy based on markdown sections.")
    # Step 1: Chunk using DocumentChunker
    for i, c in enumerate(sections):
        # Determine current section level
        current_level = c.get('level', 1)
        # Determine current chapter number
        chapter_num = c.get('chapter', 0)
        # Pop sections that are deeper or siblings
        while heading_stack and heading_stack[-1][0] >= current_level:
            heading_stack.pop()
        # Push pair of (level, heading)
        if c['heading'] != "Introduction":
            heading_stack.append((current_level, c['heading']))
        # Construct section path
        path_list = [h[1] for h in heading_stack]
        full_section_path = " ".join(path_list)
        full_section_path = f"Chapter {chapter_num} " + full_section_path

        # Use DocumentChunker to recursively split this section
        sub_chunks = chunker.chunk(c['content'])
        # Regex to find page markers like "--- Page 3 ---"
        page_pattern = re.compile(r'--- Page (\d+) ---')

        # Iterate through each chunk produced from this section
        for sub_chunk_id, sub_chunk in enumerate(sub_chunks):
            # Track all pages this specific chunk touches
            chunk_pages = set()
            # Split the sub_chunk by page markers to see if it
            # spans multiple pages.
            fragments = page_pattern.split(sub_chunk)
            # If there is content before the first page marker,
            # it belongs to the current_page.
            if fragments[0].strip():
                page_to_chunk_ids.setdefault(current_page, set()).add(total_chunks+sub_chunk_id)
                chunk_pages.add(current_page)
            # Process the new pages found within this sub_chunk. 
            # Step by 2 where each pair represents (page number, text after it)
            for i in range(1, len(fragments), 2):
                try:
                    # Get the new page number from the marker
                    new_page = int(fragments[i]) + 1
                    # If there is text after this marker, it belongs to the new_page.
                    if fragments[i+1].strip():
                        page_to_chunk_ids.setdefault(new_page, set()).add(total_chunks + sub_chunk_id)
                        chunk_pages.add(new_page)
                    current_page = new_page
                except (IndexError, ValueError):
                    continue
            # Clean sub_chunk by removing page markers
            clean_chunk = re.sub(page_pattern, '', sub_chunk).strip()
            # Skip introduction chunks for embedding
            if c["heading"] == "Introduction":
                continue
            
            # Prepare metadata
            meta = {
                "filename": markdown_file,
                "mode": chunk_config.to_string(),
                "char_len": len(clean_chunk),
                "word_len": len(clean_chunk.split()),
                "section": c['heading'],
                "section_path": full_section_path,
                "text_preview": clean_chunk[:100],
                "page_numbers": sorted(list(chunk_pages)),
                "chunk_id": total_chunks + sub_chunk_id
            }

            # Prepare chunk with prefix
            if use_headings:
                chunk_prefix = (
                    f"Description: {full_section_path} "
                    f"Content: "
                )
            else:
                chunk_prefix = ""

            all_chunks.append(chunk_prefix+clean_chunk)
            sources.append(markdown_file)
            metadata.append(meta)
        total_chunks += len(sub_chunks)

    # Convert the sets to sorted lists for a clean, predictable output
    final_map = {}
    for page, id_set in page_to_chunk_ids.items():
        final_map[page] = sorted(list(id_set))

    output_file = artifacts_dir / f"{index_prefix}_page_to_chunk_map.json"
    with open(output_file, "w") as f:
        json.dump(final_map, f, indent=2)
    print(f"Saved page to chunk ID map: {output_file}")

    # Chunking completes

    # Step 2: Create embeddings for FAISS index
    print(f"Embedding {len(all_chunks):,} chunks with {pathlib.Path(embedding_model_path).stem} ...")
    embedder = SentenceTransformer(embedding_model_path)

    if use_multiprocessing:
        print("Starting multi-process pool for embeddings...")
        # Start the pool. Adjust number of workers as needed.
        pool = embedder.start_multi_process_pool(num_workers=4)
        try:
            # Compute embeddings in parallel
            embeddings = embedder.encode_multi_process(
                all_chunks, 
                pool, 
                batch_size=32
            )
        finally:
            # Stop the pool to prevent hanging processes
            embedder.stop_multi_process_pool(pool)
    else:
        # Standard single-process embedding
        embeddings = embedder.encode(
            all_chunks, 
            batch_size=8, 
            show_progress_bar=True,
            convert_to_numpy=True 
        )

    # Step 3: Build FAISS index
    print(f"Building FAISS index for {len(all_chunks):,} chunks...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, str(artifacts_dir / f"{index_prefix}.faiss"))
    print(f"FAISS Index built successfully: {index_prefix}.faiss")

    # Step 4: Build BM25 index
    print(f"Building BM25 index for {len(all_chunks):,} chunks...")
    tokenized_chunks = [preprocess_for_bm25(chunk) for chunk in all_chunks]
    bm25_index = BM25Okapi(tokenized_chunks)
    with open(artifacts_dir / f"{index_prefix}_bm25.pkl", "wb") as f:
        pickle.dump(bm25_index, f)
    print(f"BM25 Index built successfully: {index_prefix}_bm25.pkl")

    # Step 5: Dump index artifacts
    with open(artifacts_dir / f"{index_prefix}_chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    with open(artifacts_dir / f"{index_prefix}_sources.pkl", "wb") as f:
        pickle.dump(sources, f)
    with open(artifacts_dir / f"{index_prefix}_meta.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print(f"Saved all index artifacts with prefix: {index_prefix}")

# ------------------------ Helper functions ------------------------------

def build_docling_pgvec_index(
    embedding_model_path: str,
    artifacts_dir: os.PathLike,
    index_prefix: str,
    use_multiprocessing: bool = False,
    experimental_chunking: bool = False,
    docling_json: str = None
) -> None:
    """
    Build a Docling-native pgvector index.

    pgvector is the only vector storage for this path. The pickle artifacts are
    compatibility projections for the current chat/logging code.
    """
    if not experimental_chunking:
        print("Skipping since experimental chunking is not used")
        return

    print("Using Docling pgvector indexing strategy.")
    assert docling_json is not None, "docling_json path must be provided for experimental chunking"

    chunk_result = chunk_docling_hierarchical_json(docling_json)
    records = chunk_result.records
    page_to_chunk_ids = chunk_result.page_to_chunk_ids

    all_chunks = [record.llm_text for record in records]
    sources = [record.source for record in records]
    metadata = [record.metadata for record in records]
    embedding_texts = [record.embedding_text for record in records]
    
    # Convert the sets to sorted lists for a clean, predictable output
    final_map = {}
    for page, id_set in page_to_chunk_ids.items():
        final_map[page] = sorted(list(id_set))

    exp_flag = 'exp_'
    output_file = artifacts_dir / f"{exp_flag}{index_prefix}_page_to_chunk_map.json"
    with open(output_file, "w") as f:
        json.dump(final_map, f, indent=2)
    print(f"Saved page to chunk ID map: {output_file}")

    with open(artifacts_dir / f"{exp_flag}{index_prefix}_chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    with open(artifacts_dir / f"{exp_flag}{index_prefix}_sources.pkl", "wb") as f:
        pickle.dump(sources, f)
    with open(artifacts_dir / f"{exp_flag}{index_prefix}_meta.pkl", "wb") as f:
        pickle.dump(metadata, f)
    with open(artifacts_dir / f"{exp_flag}{index_prefix}_docling_records.pkl", "wb") as f:
        pickle.dump(records, f)
    print(f"Saved Docling compatibility artifacts with prefix: {exp_flag}{index_prefix}")

    if not records:
        print("No Docling records produced; skipping pgvector write.")
        return

    print(f"Embedding {len(records):,} Docling records with {pathlib.Path(embedding_model_path).stem} ...")
    embedder = SentenceTransformer(embedding_model_path)

    if use_multiprocessing:
        print("Starting multi-process pool for embeddings...")
        pool = embedder.start_multi_process_pool(num_workers=4)
        try:
            embeddings = embedder.encode_multi_process(
                embedding_texts,
                pool, 
                batch_size=32
            )
        finally:
            # Stop the pool to prevent hanging processes
            embedder.stop_multi_process_pool(pool)
    else:
        embeddings = embedder.encode(
            embedding_texts,
            batch_size=8, 
            show_progress_bar=True,
            convert_to_numpy=True 
        )

    _write_docling_pgvector_index(
        index_prefix=exp_flag + index_prefix,
        records=records,
        embeddings=embeddings,
    )
    print(f"Saved {len(records):,} Docling records to pgvector with prefix: {exp_flag}{index_prefix}")

    # Build BM25 index as well

    print(f"Building BM25 index for {len(all_chunks):,} chunks...")
    tokenized_chunks = [preprocess_for_bm25(chunk) for chunk in all_chunks]
    bm25_index = BM25Okapi(tokenized_chunks)
    with open(artifacts_dir / f"{exp_flag}{index_prefix}_bm25.pkl", "wb") as f:
        pickle.dump(bm25_index, f)
    print(f"BM25 Index built successfully: {exp_flag}{index_prefix}_bm25.pkl")


def _get_pgvector_dsn() -> str:
    dsn = (
        os.environ.get("TOKENSMITH_PGVECTOR_DSN")
        or os.environ.get("PGVECTOR_DSN")
        or os.environ.get("DATABASE_URL")
    )
    if not dsn:
        raise RuntimeError(
            "Set TOKENSMITH_PGVECTOR_DSN, PGVECTOR_DSN, or DATABASE_URL to build the pgvector index."
        )
    return dsn


def _embedding_to_vector_literal(embedding) -> str:
    values = [float(value) for value in embedding]
    return "[" + ",".join(f"{value:.9g}" for value in values) + "]"


def _write_docling_pgvector_index(index_prefix: str, records, embeddings) -> None:
    try:
        import psycopg
        from psycopg import sql
        from psycopg.types.json import Jsonb
    except ImportError as exc:
        raise RuntimeError(
            "pgvector indexing requires psycopg. Install it with `pip install psycopg[binary]`."
        ) from exc

    if len(records) != len(embeddings):
        raise ValueError(f"Record/embedding mismatch: {len(records)} records vs {len(embeddings)} embeddings")

    dim = int(embeddings.shape[1])
    table_name = os.environ.get("TOKENSMITH_PGVECTOR_TABLE", DEFAULT_PGVECTOR_TABLE)
    table = sql.Identifier(table_name)

    create_table = sql.SQL("""
        CREATE TABLE IF NOT EXISTS {table} (
            index_prefix text NOT NULL,
            chunk_id integer NOT NULL,
            doc_id text NOT NULL,
            source text NOT NULL,
            content text NOT NULL,
            embedding_text text NOT NULL,
            llm_text text NOT NULL,
            section text,
            section_path text,
            headings text[],
            page_numbers integer[],
            node_ids text[],
            doc_item_refs text[],
            doc_item_labels text[],
            nodes jsonb NOT NULL,
            doc_items jsonb NOT NULL,
            relationships jsonb NOT NULL,
            metadata jsonb NOT NULL,
            embedding vector({dim}) NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (index_prefix, chunk_id)
        )
    """).format(table=table, dim=sql.SQL(str(dim)))

    insert_record = sql.SQL("""
        INSERT INTO {table} (
            index_prefix,
            chunk_id,
            doc_id,
            source,
            content,
            embedding_text,
            llm_text,
            section,
            section_path,
            headings,
            page_numbers,
            node_ids,
            doc_item_refs,
            doc_item_labels,
            nodes,
            doc_items,
            relationships,
            metadata,
            embedding
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::vector
        )
        ON CONFLICT (index_prefix, chunk_id) DO UPDATE SET
            doc_id = EXCLUDED.doc_id,
            source = EXCLUDED.source,
            content = EXCLUDED.content,
            embedding_text = EXCLUDED.embedding_text,
            llm_text = EXCLUDED.llm_text,
            section = EXCLUDED.section,
            section_path = EXCLUDED.section_path,
            headings = EXCLUDED.headings,
            page_numbers = EXCLUDED.page_numbers,
            node_ids = EXCLUDED.node_ids,
            doc_item_refs = EXCLUDED.doc_item_refs,
            doc_item_labels = EXCLUDED.doc_item_labels,
            nodes = EXCLUDED.nodes,
            doc_items = EXCLUDED.doc_items,
            relationships = EXCLUDED.relationships,
            metadata = EXCLUDED.metadata,
            embedding = EXCLUDED.embedding,
            created_at = now()
    """).format(table=table)

    index_name = sql.Identifier(f"{table_name}_embedding_hnsw_l2")
    create_vector_index = sql.SQL("""
        CREATE INDEX IF NOT EXISTS {index_name}
        ON {table}
        USING hnsw (embedding vector_l2_ops)
    """).format(index_name=index_name, table=table)

    with psycopg.connect(_get_pgvector_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(create_table)
            cur.execute(sql.SQL("DELETE FROM {table} WHERE index_prefix = %s").format(table=table), (index_prefix,))
            for record, embedding in zip(records, embeddings):
                cur.execute(
                    insert_record,
                    (
                        index_prefix,
                        record.chunk_id,
                        record.doc_id,
                        record.source,
                        record.text,
                        record.embedding_text,
                        record.llm_text,
                        record.section,
                        record.section_path,
                        record.headings,
                        record.page_numbers,
                        record.node_ids,
                        record.doc_item_refs,
                        record.doc_item_labels,
                        Jsonb(record.nodes),
                        Jsonb(record.doc_items),
                        Jsonb(record.relationships),
                        Jsonb(asdict(record)),
                        _embedding_to_vector_literal(embedding),
                    ),
                )
            if dim <= 2000:
                cur.execute(create_vector_index)
            else:
                print(
                    f"Skipping pgvector HNSW index because embedding dimension {dim} exceeds "
                    "pgvector's 2000-dimension HNSW limit. Retrieval will use exact scan."
                )

def preprocess_for_bm25(text: str) -> list[str]:
    """
    Simplifies text to keep only letters, numbers, underscores, hyphens,
    apostrophes, plus, and hash — suitable for BM25 tokenization.
    """
    # Convert to lowercase
    text = text.lower()

    # Keep only allowed characters
    text = re.sub(r"[^a-z0-9_'#+-]", " ", text)

    # Split by whitespace
    tokens = text.split()

    return tokens
