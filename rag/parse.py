from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Literal

import pdfplumber
import pymupdf


@dataclass
class Element:
    element_id: str
    type: Literal["paragraph", "title", "table", "list_item"]
    page: int
    bbox: list[float]
    text: str
    table_md: str = ""
    table_flat: str = ""


def _cluster_columns(words: list[dict], x_threshold: float = 20.0) -> list[list[dict]]:
    """Group words into columns by x0 clustering."""
    if not words:
        return []
    sorted_words = sorted(words, key=lambda w: w["x0"])
    columns: list[list[dict]] = [[sorted_words[0]]]
    for w in sorted_words[1:]:
        if w["x0"] - columns[-1][-1]["x0"] > x_threshold:
            columns.append([w])
        else:
            columns[-1].append(w)
    return columns


def _order_words_reading(words: list[dict], x_threshold: float = 20.0) -> str:
    """Reconstruct reading order from PDF text blocks (handles multi-column)."""
    columns = _cluster_columns(words, x_threshold)
    # Sort each column top-to-bottom, then concatenate left-to-right
    ordered: list[str] = []
    for col in columns:
        col.sort(key=lambda w: w["y0"])
        ordered.append(" ".join(w.get("text", "") for w in col))
    return " ".join(ordered)


def parse_pdf(file_bytes: bytes) -> list[Element]:
    """Parse a PDF into structured elements using PyMuPDF + pdfplumber."""
    elements: list[Element] = []
    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    pdf = pdfplumber.open(io.BytesIO(file_bytes))

    for page_num in range(len(doc)):
        page = doc[page_num]
        block_id = 0

        # --- Text blocks via PyMuPDF ---
        blocks = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)["blocks"]
        words_on_page: list[dict] = []
        for b in blocks:
            if b["type"] == 0:  # text block
                for line in b.get("lines", []):
                    for span in line.get("spans", []):
                        words_on_page.append({
                            "text": span["text"],
                            "x0": span["bbox"][0],
                            "y0": span["bbox"][1],
                            "x1": span["bbox"][2],
                            "y1": span["bbox"][3],
                            "font": span.get("font", ""),
                            "size": span.get("size", 0),
                        })

        # Reconstruct reading order for this page
        page_text = _order_words_reading(words_on_page)
        if page_text.strip():
            elements.append(Element(
                element_id=f"p{page_num}_b{block_id}",
                type="paragraph",
                page=page_num + 1,
                bbox=[0, 0, 0, 0],
                text=page_text.strip(),
            ))
            block_id += 1

        # --- Tables via pdfplumber ---
        try:
            plumb_page = pdf.pages[page_num]
            for table_idx, table in enumerate(plumb_page.extract_tables()):
                if not table or not table[0]:
                    continue
                # Markdown rendering
                header = table[0]
                md_lines = ["| " + " | ".join(str(c or "") for c in header) + " |"]
                md_lines.append("| " + " | ".join("---" for _ in header) + " |")
                for row in table[1:]:
                    md_lines.append("| " + " | ".join(str(c or "") for c in row) + " |")
                md = "\n".join(md_lines)

                # Flattened NL rendering: "row_header, col_header = value"
                flat_parts: list[str] = []
                if len(table) > 1:
                    col_headers = [str(c or "") for c in table[0]]
                    for row in table[1:]:
                        row_label = str(row[0] or "")
                        for ci, cell in enumerate(row[1:], 1):
                            if cell and ci < len(col_headers):
                                flat_parts.append(f"{row_label}, {col_headers[ci]} = {cell}")
                flat = "; ".join(flat_parts)

                elements.append(Element(
                    element_id=f"p{page_num}_t{table_idx}",
                    type="table",
                    page=page_num + 1,
                    bbox=[0, 0, 0, 0],
                    text=md,
                    table_md=md,
                    table_flat=flat,
                ))
        except Exception:
            pass  # tables are best-effort

    doc.close()
    pdf.close()
    return elements
