"""MCP server: lets an agent browse analysed papers and submit new ones.

Run over stdio (default, for local agents) or ``--http`` (streamable HTTP). In HTTP mode the
server refuses to read local file paths, since remote callers must not read the host's disk.
"""
from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from mcp.server.fastmcp import FastMCP

from . import report
from .analysis import Analyzer, mermaid
from .config import load_config
from .ingest import IngestError
from .pipeline import Pipeline, build, decode_b64, safe_filename
from .store import Project, read_json

INSTRUCTIONS = """Backtrack analyses academic papers for total understanding.
Use list_papers/search_papers to find analysed papers, get_analysis for a section
(overview, structure, related, position, prerequisites, learning_path, missing, report),
explain_paper for a level-specific explanation, and ask_paper for questions grounded in the paper.
Use upload_paper to submit a new paper (path, URL, DOI, arXiv id, or base64 content); processing runs
in the background, so poll get_status. Cited works that cannot be fetched from public sources are
listed by get_missing; after a human supplies PDFs, call add_missing_pdf or rescan_paper."""

mcp = FastMCP("backtrack", instructions=INSTRUCTIONS)
_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="backtrack-job")
_lock = threading.Lock()
_allow_local_paths = True

Section = Literal["overview", "structure", "related", "position", "prerequisites", "learning_path", "missing", "report"]
STAGE_FOR = {"overview": "explain", "structure": "structure", "related": "related", "position": "position",
             "prerequisites": "prerequisites", "learning_path": "learning_path"}


def _pipe(require_llm: bool = False) -> Pipeline:
    return build(load_config(), require_llm=require_llm)


def _project(paper_id: str) -> tuple[Pipeline, Project]:
    pipe = _pipe()
    return pipe, pipe.ws.get(paper_id)


@mcp.tool()
def list_papers() -> list[dict]:
    """List every paper in the workspace with its processing state and counts of related/missing works."""
    return [p.summary() for p in _pipe().ws.projects()]


@mcp.tool()
def search_papers(query: str) -> list[dict]:
    """Find analysed papers whose title, authors, abstract, TL;DR or prerequisite concepts match all query terms."""
    return _pipe().ws.search(query)


@mcp.tool()
def get_status(paper_id: str) -> dict:
    """Processing state of a paper: queued, running (with message), fetched, done or error."""
    _, proj = _project(paper_id)
    return {**proj.summary(), "status": proj.status}


@mcp.tool()
def get_paper(paper_id: str) -> dict:
    """Bibliographic metadata of the primary paper, plus its one-line summary if analysed."""
    _, proj = _project(paper_id)
    ex = proj.load_stage("explain") or {}
    meta = {k: v for k, v in proj.meta.items() if k not in ("reference_ids", "reference_dois")}
    return {**meta, "tldr": ex.get("tldr"), "contribution": ex.get("contribution"), "state": proj.status.get("state")}


@mcp.tool()
def get_analysis(paper_id: str, section: Section = "overview") -> dict | str:
    """Fetch one analysis section: overview (levels, TL;DR), structure (claims, methods, limits, glossary),
    related (cited works + how each relates + landscape), position (the paper's role in its field),
    prerequisites (concept dependency graph), learning_path (1h/1w/1m/1y plans), missing (works to
    fetch manually), or report (full Markdown)."""
    _, proj = _project(paper_id)
    if section == "report":
        return proj.report_path.read_text() if proj.report_path.exists() else "Report not generated yet."
    if section == "missing":
        return {"missing": read_json(proj.root / "missing.json", [])}
    data = proj.load_stage(STAGE_FOR[section])
    if data is None:
        raise ValueError(f"section {section!r} is not available yet (state: {proj.status.get('state')})")
    return data


@mcp.tool()
def explain_paper(paper_id: str, level: Literal["layperson", "undergraduate", "graduate", "expert"] = "undergraduate") -> str:
    """Explanation of the paper pitched at the given expertise level."""
    _, proj = _project(paper_id)
    ex = proj.load_stage("explain")
    if not ex:
        raise ValueError("paper has not been analysed yet")
    return ex["levels"][level]


@mcp.tool()
def get_learning_path(paper_id: str, horizon: Literal["1h", "1w", "1m", "1y"] = "1w") -> dict:
    """Study plan for the prerequisites of the paper at a time horizon: 1 hour, 1 week, 1 month or 1 year."""
    _, proj = _project(paper_id)
    paths = proj.load_stage("learning_path")
    if not paths:
        raise ValueError("learning paths not available yet")
    return paths[horizon]


@mcp.tool()
def get_prerequisite_graph(paper_id: str, format: Literal["json", "mermaid"] = "json") -> dict | str:
    """Prerequisite concepts and their dependencies, as JSON (nodes in study order) or a Mermaid diagram."""
    _, proj = _project(paper_id)
    g = proj.load_stage("prerequisites")
    if not g:
        raise ValueError("prerequisites not available yet")
    return mermaid(g) if format == "mermaid" else g


@mcp.tool()
def get_missing(paper_id: str) -> dict:
    """Cited works that could not be retrieved from public sources, with links, and where to drop the PDFs."""
    _, proj = _project(paper_id)
    return {"missing": read_json(proj.root / "missing.json", []), "dropbox": str(proj.dropbox_dir),
            "instructions": "Place PDFs in the dropbox folder (any filename), then call rescan_paper."}


@mcp.tool()
def ask_paper(paper_id: str, question: str) -> str:
    """Ask a question about a paper; answered by the configured LLM from the paper text and analysis."""
    pipe = _pipe(require_llm=True)
    return Analyzer(pipe.llm, pipe.ws.get(paper_id)).ask(question)


def _run_job(pipe: Pipeline, proj: Project, depth: int | None, analyze: bool) -> None:
    try:
        pipe.process(proj, depth=depth, analyze=analyze)
    except Exception:
        pass  # process() records the error in status.json


@mcp.tool()
def upload_paper(source: str = "", content_base64: str = "", filename: str = "paper.pdf",
                 depth: int | None = None, analyze: bool = True, wait: bool = False) -> dict:
    """Submit a paper for analysis. Give `source` (a URL, DOI, arXiv id, or - in stdio mode - a local
    file path) or `content_base64` with `filename`. Processing (citation crawl to `depth`, then LLM
    analysis) runs in the background; poll get_status with the returned paper_id, or set wait=true."""
    pipe = _pipe(require_llm=analyze)
    if source and not content_base64 and not _allow_local_paths and Path(source).expanduser().exists():
        raise ValueError("local paths are disabled on this server; send content_base64 instead")
    try:
        if content_base64:
            proj = pipe.create_project(content=decode_b64(content_base64), filename=safe_filename(filename))
        else:
            proj = pipe.create_project(source)
    except IngestError as e:
        raise ValueError(str(e)) from e
    fut = _pool.submit(_run_job, pipe, proj, depth, analyze)
    if wait:
        fut.result()
    return {"paper_id": proj.id, **proj.summary(), "status": proj.status}


@mcp.tool()
def add_missing_pdf(paper_id: str, filename: str, content_base64: str = "", path: str = "") -> dict:
    """Supply a PDF for a missing cited work (base64 content, or a local path in stdio mode).
    It is matched against the missing list by filename, DOI or title, and the report is refreshed."""
    pipe, proj = _project(paper_id)
    if content_base64:
        data = decode_b64(content_base64)
    elif path and _allow_local_paths:
        data = Path(path).expanduser().read_bytes()
    else:
        raise ValueError("provide content_base64 (or path, in stdio mode)")
    (proj.dropbox_dir / safe_filename(filename)).write_bytes(data)
    return pipe.rescan(proj)


@mcp.tool()
def rescan_paper(paper_id: str, reanalyze: bool = False) -> dict:
    """Pick up PDFs placed in the paper's dropbox folder and update the missing list. With reanalyze=true,
    regenerate the related-work, position, prerequisite and learning-path analyses using the new material."""
    pipe = _pipe(require_llm=reanalyze)
    return pipe.rescan(pipe.ws.get(paper_id), reanalyze=reanalyze)


@mcp.resource("backtrack://papers")
def papers_resource() -> list[dict]:
    """Index of all papers."""
    return list_papers()


@mcp.resource("backtrack://paper/{paper_id}/report")
def report_resource(paper_id: str) -> str:
    """Full Markdown report for one paper."""
    _, proj = _project(paper_id)
    return proj.report_path.read_text() if proj.report_path.exists() else report.render(proj)


def main(argv: list[str] | None = None) -> None:
    global _allow_local_paths
    argv = sys.argv[1:] if argv is None else argv
    if "--http" in argv:
        _allow_local_paths = False
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
