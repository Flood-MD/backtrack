"""Turn an uploaded file into text, best-guess metadata and a reference list."""
from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

from .sources import DOI_RE, clean_arxiv, clean_doi

REF_HEADING = re.compile(
    r"^\s*(?:\d+\.?\s*)?(references|bibliography|works cited|literature cited)\s*$", re.I | re.M
)


class IngestError(ValueError):
    pass


def is_pdf(data: bytes) -> bool:
    return data.lstrip()[:5].startswith(b"%PDF")


def extract_pdf_text(path: Path, max_pages: int = 400) -> tuple[str, dict]:
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            reader.decrypt("")
        pages = []
        for page in reader.pages[:max_pages]:
            try:
                pages.append(page.extract_text() or "")
            except Exception:  # a single bad page should not sink the paper
                pages.append("")
        info = reader.metadata or {}
        meta = {"pdf_title": str(info.get("/Title") or ""), "pdf_author": str(info.get("/Author") or "")}
    except Exception as e:
        raise IngestError(f"could not read PDF: {e}") from e
    text = "\n\n".join(pages).strip()
    if len(text) < 200:
        raise IngestError(
            "no extractable text (scanned PDF?). Run OCR (e.g. `ocrmypdf`) and upload again."
        )
    return text, meta


def load_document(path: Path) -> tuple[str, dict]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return extract_pdf_text(path)
    if ext in (".txt", ".md", ".tex"):
        return path.read_text(errors="replace"), {}
    raise IngestError(f"unsupported file type {ext!r} (use .pdf, .txt, .md)")


def split_references(text: str) -> tuple[str, str]:
    """Return (body, references_section). Uses the last 'References' heading."""
    matches = list(REF_HEADING.finditer(text))
    if not matches:
        return text, ""
    m = matches[-1]
    # Guard against a heading in a table of contents near the start.
    if m.start() < len(text) * 0.3:
        return text, ""
    tail = text[m.end():]
    # Appendices sometimes follow references; stop at one.
    app = re.search(r"^\s*(appendix|appendices|supplementary)\b.*$", tail, re.I | re.M)
    if app:
        tail = tail[: app.start()]
    return text[: m.start()], tail.strip()


def guess_identifiers(text: str, pdf_meta: dict | None = None) -> dict:
    head = text[:6000]
    out: dict = {}
    if doi := clean_doi(head):
        out["doi"] = doi
    m = re.search(r"arxiv[:\s]*(\d{4}\.\d{4,5})(?:v\d+)?", head, re.I)
    if m:
        out["arxiv_id"] = m.group(1)
    title = (pdf_meta or {}).get("pdf_title", "").strip()
    if len(title) < 8 or title.lower().startswith(("microsoft word", "untitled")) or title.endswith((".pdf", ".tex", ".dvi")):
        title = ""
    if not title:
        lines = [l.strip() for l in head.splitlines() if 15 <= len(l.strip()) <= 200]
        title = lines[0] if lines else ""
    out["title"] = title
    return out


def split_reference_entries(ref_text: str) -> list[str]:
    """Split a references section into individual entry strings (heuristic)."""
    if not ref_text:
        return []
    t = ref_text.replace("\r", "")
    # Numbered styles: "[12] ..." or "12. ..." at line start.
    parts = re.split(r"\n\s*(?=\[\d{1,3}\]\s)", "\n" + t)
    if len(parts) < 3:
        parts = re.split(r"\n\s*(?=\d{1,3}\.\s+[A-Z])", "\n" + t)
    if len(parts) < 3:
        # Author-year: a new entry starts after a line ending in '.' followed by a capitalised surname + comma.
        parts = re.split(r"(?<=[.\d])\n(?=[A-Z][A-Za-z'\-]+,\s+[A-Z]\.)", t)
    entries = [re.sub(r"\s+", " ", p).strip() for p in parts]
    entries = [re.sub(r"^\[\d+\]\s*|^\d+\.\s+", "", e) for e in entries]
    return [e for e in entries if len(e) > 25]


def entry_doi(entry: str) -> str | None:
    m = DOI_RE.search(entry)
    return clean_doi(m.group(1)) if m else None


def entry_arxiv(entry: str) -> str | None:
    m = re.search(r"arxiv[:\s]*(\d{4}\.\d{4,5})", entry, re.I)
    return m.group(1) if m else clean_arxiv(entry)


def excerpt_for_llm(text: str, max_chars: int) -> tuple[str, bool]:
    """Keep the head and tail when text exceeds the budget. Returns (text, truncated)."""
    if len(text) <= max_chars:
        return text, False
    head = int(max_chars * 0.65)
    tail = max_chars - head
    return text[:head] + "\n\n[... middle of paper omitted for length ...]\n\n" + text[-tail:], True
