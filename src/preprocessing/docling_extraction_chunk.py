from pathlib import Path
import uuid
import json
import re
from typing import Dict, Iterator, List, Optional, Sequence
import sys
from dataclasses import dataclass
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption, InputFormat
from docling.backend.docling_parse_backend import DoclingParseDocumentBackend

# LlmaIndex and Docling dependencies
from docling_core.types.doc import DoclingDocument
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.core.schema import TextNode
from llama_index.core import Document as LIDocument
from llama_index.core.ingestion.pipeline import run_transformations

def _uuid4_doc_id_gen(doc: DoclingDocument, file_path: str | Path) -> str:
    return str(uuid.uuid4())

def exp_extract(args, project_root, pdfs, output_prefix, *, use_ocr: bool = False, use_table_structure: bool = False):
    """
    All experimental extraction happens here. This isolates from existing run-extract routine

    This is invoked by extraction.py

    Args:
    - args: List of additional CLI arguments for the experiment
    - project_root: Root directory of the project
    - pdfs: List of PDF files to process
    - output_prefix: Prefix for the output files
    """
    print("Entered experimental extraction with the following CLI args: ", args)
    load_ir = "load_existing_ir" in args
    for pdf_path in pdfs:
        pdf_name = pdf_path.stem
        output_dir = Path("data/exp")
        if not load_ir:
            print(f"Generating Docling IR for '{pdf_path}'...")
            generate_docling_ir(
                str(pdf_path),
                str(output_dir),
                use_ocr=use_ocr,
                use_table_structure=use_table_structure,
            )
        chunks = generate_docling_hierarchical_json(
            pdf_name,
            str(output_dir / "EXP_IR.json"))
        with open(str(output_dir / f"{output_prefix}{pdf_name}--hierarchical.json"), "w") as f:
            json.dump([chunk.model_dump() for chunk in chunks], f, indent=4, ensure_ascii=False)
        print(f"Saved hierarchical JSON for '{pdf_path}' to {output_dir / f'{output_prefix}{pdf_name}--hierarchical.json'}")

def generate_docling_ir(input_file_path, output_file_path, *, use_ocr: bool = False, use_table_structure: bool = False):
    """
    Instead of flattening to Markdown after Docling conversion directly, preserve the IR
DoclingParseDocumentBackend
    Sample further directions:
    - Convert to markdown/json with more metadata. Currently json only knows heading, chapter, and 1st sub-chapter
    - Skip json and directly chunk using the richer IR.

    Args:
        input_file_path (str): The path to the source file (e.g., "/path/to/file.pdf").
        output_file_path (str): The path to the destination file
    """
    source = Path(input_file_path)
    if not source.exists():
        print(f"Error: Input file not found at {input_file_path}", file=sys.stderr)
        return

    # Disable OCR and table structure extraction for faster processing
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = use_ocr
    pipeline_options.do_table_structure = use_table_structure

    converter = DocumentConverter(
    format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options, backend=DoclingParseDocumentBackend)
        }
    )
    
    try:
        # Convert the entire document once
        result = converter.convert(source)
    except Exception as e:
        print(f"Error during conversion: {e}", file=sys.stderr)
        return
        
    doc = result.document

    doc.save_as_json(f"{output_file_path}/EXP_IR.json")

def generate_docling_hierarchical_json(src_name, input_file_path):
    """
    Generates chunks from a Docling IR using Docling HierarchicalChunker (implemented by LlamaIndex)

    LlamaIndex has a template support for Docling documents. It is aware of the rich metadata 
    a Docling IR holds. This function builds on top of this support to maintain structural awareness,
    instead of flattening the stateful IR to markdown too early.

    Args:
        src_name (str): A name for the source document, used in metadata.
        input_file_path (str): The path to the source file.
    """
    docling_ir = DoclingDocument.load_from_json(input_file_path)
    # TODO: This is Docling-aware parser from LlamaIndex. There is also SentenceSplitter that could be helpful
    node_parser = DoclingNodeParser()
    llamaIndex_doc = LIDocument(
        doc_id=_uuid4_doc_id_gen(docling_ir, src_name),
        text=json.dumps(docling_ir.export_to_dict())
    )
    chunks = run_transformations([llamaIndex_doc], [node_parser], show_progress=True)
    return chunks

@dataclass
class DoclingChunkRecord:
    chunk_id: int
    doc_id: str
    source: str

    text: str
    embedding_text: str
    llm_text: str

    section: str
    section_path: str
    headings: list[str]

    page_numbers: list[int]
    node_ids: list[str]
    doc_item_refs: list[str]
    doc_item_labels: list[str]

    nodes: list[dict]
    doc_items: list[dict]
    relationships: dict

    char_len: int
    word_len: int
    token_count: Optional[int]

    metadata: dict


@dataclass
class DoclingChunkingResult:
    records: list[DoclingChunkRecord]
    page_to_chunk_ids: dict[int, set[int]]

    def as_legacy_tuple(self):
        all_chunks = [record.llm_text for record in self.records]
        sources = [record.source for record in self.records]
        metadata = [record.metadata for record in self.records]
        return all_chunks, sources, metadata, self.page_to_chunk_ids

    def __iter__(self) -> Iterator:
        """Support existing tuple-unpacking callers during the migration."""
        return iter(self.as_legacy_tuple())


def chunk_docling_hierarchical_json(input_file_path):
    """
    Build structured chunks from LlamaIndex Docling nodes.

    The returned records are the canonical representation for Docling-backed
    retrieval. Legacy text chunks, sources, and metadata can be derived from
    the result for the current FAISS/BM25 pipeline.

    Args:
        input_file_path: Path, or single-item path list, produced by
            generate_docling_hierarchical_json().
    """
    def first_input_path(value) -> Path:
        if isinstance(value, (str, Path)):
            return Path(value)
        if isinstance(value, Sequence) and value:
            return Path(value[0])
        raise ValueError("input_file_path must be a path or a non-empty sequence of paths")

    source_path = first_input_path(input_file_path)
    with open(source_path, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    records: List[DoclingChunkRecord] = []
    page_to_chunk_ids: Dict[int, set[int]] = {}

    max_chars_per_chunk = 1800
    min_chunk_chars = 120
    current_chunk = None
    seen_main_content = False

    # FIXME: hardcoded to textbook
    front_matter_markers = (
        "database system concepts",
        "about the authors",
        "published by mcgraw-hill",
        "cover image",
        "copyright",
        "dedication",
    )

    # FIXME: hardcoded to textbook
    back_matter_markers = (
        "index",
        "further reading",
    )

    def normalize_text(value: str) -> str:
        return " ".join(value.lower().split())

    def extract_page_numbers(doc_items: List[Dict]) -> List[int]:
        pages = {
            prov.get("page_no")
            for item in doc_items
            for prov in item.get("prov", [])
            if prov.get("page_no") is not None
        }
        return sorted(pages)

    def extract_doc_item_refs(doc_items: List[Dict]) -> List[str]:
        refs = {
            item.get("self_ref")
            for item in doc_items
            if item.get("self_ref")
        }
        return sorted(refs)

    def render_embedding_text(section_path: str, chunk_text: str) -> str:
        if not section_path:
            return chunk_text
        return f"Section: {section_path}\nContent: {chunk_text}"

    def render_llm_text(section_path: str, page_numbers: List[int], chunk_text: str) -> str:
        parts = []
        if section_path:
            parts.append(f"Section: {section_path}")
        if page_numbers:
            page_list = ", ".join(str(page) for page in page_numbers)
            parts.append(f"Pages: {page_list}")
        parts.append(chunk_text)
        return "\n".join(parts)

    def is_main_content_heading(headings: List[str]) -> bool:
        if not headings:
            return False
        heading = headings[-1].strip()
        return bool(
            re.match(r"^chapter\s+\d+\b", heading, re.IGNORECASE)
            or re.match(r"^\d+(\.\d+)+\b", heading)
        )

    def is_noise_node(headings: List[str], text: str, page_numbers: List[int]) -> bool:
        normalized_heading = normalize_text(" > ".join(headings))
        normalized_text = normalize_text(text)

        if not seen_main_content and not is_main_content_heading(headings):
            return True

        if any(marker in normalized_heading for marker in front_matter_markers):
            return True

        if any(marker in normalized_heading for marker in back_matter_markers) and page_numbers and page_numbers[0] > 2000:
            return True

        # Skip obvious table-of-contents lines like "1.2 Purpose of Database Systems 5"
        if re.fullmatch(r"(\d+(\.\d+)+\s+.+?\s+\d+\s*)+", text.strip()):
            return True

        # Skip tiny fragments that usually come from cover/copyright noise
        word_count = len(text.split())
        if len(text) < min_chunk_chars and word_count < 20 and not re.search(r"[.!?]", text):
            return True

        return False

    def flush_current_chunk() -> None:
        nonlocal current_chunk
        if current_chunk is None:
            return

        chunk_id = len(records)
        chunk_text = current_chunk["text"].strip()
        if not chunk_text:
            current_chunk = None
            return

        section_path = current_chunk["section_path"]
        page_numbers = current_chunk["page_numbers"]
        embedding_text = render_embedding_text(section_path, chunk_text)
        llm_text = render_llm_text(section_path, page_numbers, chunk_text)
        doc_items = current_chunk["doc_items"]
        doc_item_refs = extract_doc_item_refs(doc_items)
        doc_item_labels = sorted(current_chunk["doc_item_labels"])
        node_ids = current_chunk["node_ids"]

        chunk_meta = {
            "filename": current_chunk["source"],
            "mode": "docling_hierarchical",
            "char_len": len(chunk_text),
            "word_len": len(chunk_text.split()),
            "section": current_chunk["section"],
            "section_path": section_path,
            "headings": current_chunk["headings"],
            "text_preview": chunk_text[:100],
            "page_numbers": page_numbers,
            "chunk_id": chunk_id,
            "doc_id": current_chunk["doc_id"],
            "node_ids": node_ids,
            "doc_item_refs": doc_item_refs,
            "doc_item_labels": doc_item_labels,
        }

        records.append(
            DoclingChunkRecord(
                chunk_id=chunk_id,
                doc_id=current_chunk["doc_id"],
                source=current_chunk["source"],
                text=chunk_text,
                embedding_text=embedding_text,
                llm_text=llm_text,
                section=current_chunk["section"],
                section_path=section_path,
                headings=current_chunk["headings"],
                page_numbers=page_numbers,
                node_ids=node_ids,
                doc_item_refs=doc_item_refs,
                doc_item_labels=doc_item_labels,
                nodes=current_chunk["nodes"],
                doc_items=doc_items,
                relationships=current_chunk["relationships"],
                char_len=len(chunk_text),
                word_len=len(chunk_text.split()),
                token_count=None,
                metadata=chunk_meta,
            )
        )

        for page_no in page_numbers:
            page_to_chunk_ids.setdefault(page_no, set()).add(chunk_id)

        current_chunk = None

    for node in nodes:
        text = (node.get("text") or "").strip()
        if not text:
            continue

        node_metadata = node.get("metadata") or {}
        headings = node_metadata.get("headings") or []
        doc_items = node_metadata.get("doc_items") or []
        relationships = node.get("relationships") or {}
        page_numbers = extract_page_numbers(doc_items)
        if is_main_content_heading(headings):
            seen_main_content = True

        if is_noise_node(headings, text, page_numbers):
            continue

        source_name = node_metadata.get("origin", {}).get("filename", str(source_path))
        doc_id = str(node_metadata.get("origin", {}).get("binary_hash") or source_name)
        section = headings[-1] if headings else "Unknown"
        section_path = " > ".join(headings) if headings else section
        doc_item_labels = {
            item.get("label")
            for item in doc_items
            if item.get("label")
        }
        node_id = node.get("id_")

        should_merge = (
            current_chunk is not None
            and current_chunk["section_path"] == section_path
            and (
                current_chunk["page_numbers"] == page_numbers
                or (
                    current_chunk["page_numbers"]
                    and page_numbers
                    and page_numbers[0] - current_chunk["page_numbers"][-1] <= 1
                )
            )
            and len(current_chunk["text"]) + 1 + len(text) <= max_chars_per_chunk
        )

        if not should_merge:
            flush_current_chunk()
            chunk_relationships = {node_id: relationships} if node_id else {}
            current_chunk = {
                "text": text,
                "doc_id": doc_id,
                "section": section,
                "section_path": section_path,
                "headings": headings,
                "page_numbers": page_numbers,
                "source": source_name,
                "node_ids": [node_id] if node_id else [],
                "doc_item_labels": set(doc_item_labels),
                "nodes": [node],
                "doc_items": list(doc_items),
                "relationships": chunk_relationships,
            }
            continue

        current_chunk["text"] += "\n" + text
        current_chunk["page_numbers"] = sorted(set(current_chunk["page_numbers"]).union(page_numbers))
        current_chunk["node_ids"].extend([node_id] if node_id else [])
        current_chunk["doc_item_labels"].update(doc_item_labels)
        current_chunk["nodes"].append(node)
        current_chunk["doc_items"].extend(doc_items)
        if node_id:
            current_chunk["relationships"][node_id] = relationships

    flush_current_chunk()

    return DoclingChunkingResult(
        records=records,
        page_to_chunk_ids=page_to_chunk_ids,
    )
