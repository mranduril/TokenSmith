from pathlib import Path
import uuid
import json
from typing import List, Dict
import sys
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption, InputFormat
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend

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
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options, backend=DoclingParseV2DocumentBackend)
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

def chunk_docling_hierarchical_json(input_file_path):
    """
    Alternative chunking approach that directly chunks the Docling IR without converting to markdown first.
    This is more experimental and may require custom chunking logic to fully leverage the IR's structure.

    Args:
        input_file_path List[str]: The path to the source file, a json produced by generate_docling_hierarchical_json().
    """
    # TODO: As TokenSmith current stage, we also look at only 1 input file

    with open(input_file_path[0], "r", encoding="utf-8") as f:
        nodes = json.load(f)

    all_chunks: List[str] = []
    sources: List[str] = []
    metadata: List[Dict] = []
    page_to_chunk_ids: Dict[int, set[int]] = {}

    max_chars_per_chunk = 1800
    current_chunk = None

    def extract_page_numbers(doc_items: List[Dict]) -> List[int]:
        pages = {
            prov.get("page_no")
            for item in doc_items
            for prov in item.get("prov", [])
            if prov.get("page_no") is not None
        }
        return sorted(pages)

    def flush_current_chunk() -> None:
        nonlocal current_chunk
        if current_chunk is None:
            return

        chunk_id = len(all_chunks)
        chunk_text = current_chunk["text"].strip()
        if not chunk_text:
            current_chunk = None
            return

        section_path = current_chunk["section_path"]
        prefixed_text = (
            f"Description: {section_path} Content: {chunk_text}"
            if section_path
            else chunk_text
        )

        chunk_meta = {
            "filename": current_chunk["source"],
            "mode": "docling_hierarchical",
            "char_len": len(chunk_text),
            "word_len": len(chunk_text.split()),
            "section": current_chunk["section"],
            "section_path": section_path,
            "text_preview": chunk_text[:100],
            "page_numbers": current_chunk["page_numbers"],
            "chunk_id": chunk_id,
            "node_ids": current_chunk["node_ids"],
            "doc_item_labels": sorted(current_chunk["doc_item_labels"]),
        }

        all_chunks.append(prefixed_text)
        sources.append(current_chunk["source"])
        metadata.append(chunk_meta)

        for page_no in current_chunk["page_numbers"]:
            page_to_chunk_ids.setdefault(page_no, set()).add(chunk_id)

        current_chunk = None

    for node in nodes:
        text = (node.get("text") or "").strip()
        if not text:
            continue

        node_metadata = node.get("metadata") or {}
        headings = node_metadata.get("headings") or []
        doc_items = node_metadata.get("doc_items") or []
        page_numbers = extract_page_numbers(doc_items)
        source_name = node_metadata.get("origin", {}).get("filename", str(input_file_path))
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
            and current_chunk["page_numbers"] == page_numbers
            and len(current_chunk["text"]) + 1 + len(text) <= max_chars_per_chunk
        )

        if not should_merge:
            flush_current_chunk()
            current_chunk = {
                "text": text,
                "section": section,
                "section_path": section_path,
                "page_numbers": page_numbers,
                "source": source_name,
                "node_ids": [node_id] if node_id else [],
                "doc_item_labels": set(doc_item_labels),
            }
            continue

        current_chunk["text"] += "\n" + text
        current_chunk["node_ids"].extend([node_id] if node_id else [])
        current_chunk["doc_item_labels"].update(doc_item_labels)

    flush_current_chunk()

    return all_chunks, sources, metadata, page_to_chunk_ids

