"""End-to-end orchestration: ingest -> resolve -> crawl -> analyse -> report."""
from __future__ import annotations

import base64
import hashlib
import re
from collections.abc import Callable
from pathlib import Path

from . import library, report
from .analysis import Analyzer
from .config import Config
from .ingest import (IngestError, guess_identifiers, is_pdf, load_document,
                     split_reference_entries, split_references)
from .llm import LLMClient
from .sources import Scholar, Work, clean_arxiv, clean_doi, slug
from .store import Project, Workspace

Progress = Callable[[str], None]


class Pipeline:
    def __init__(self, cfg: Config, workspace: Workspace, scholar: Scholar,
                 llm: LLMClient | None = None):
        self.cfg, self.ws, self.scholar, self.llm = cfg, workspace, scholar, llm

    # -- step 1: get the paper into a project --------------------------------
    def create_project(self, source: str | Path | None = None, *, content: bytes | None = None,
                       filename: str = "paper.pdf") -> Project:
        """``source`` is a local path, an http(s) URL, a DOI, or an arXiv id."""
        data, name = self._acquire(source, content, filename)
        ext = Path(name).suffix.lower() or ".pdf"
        if ext == ".pdf" and not is_pdf(data[:16]):
            raise IngestError("file is not a valid PDF")
        tmp_slug = slug(Path(name).stem, 40) or hashlib.sha1(data).hexdigest()[:8]
        proj = self.ws.create(tmp_slug)
        (proj.primary_dir / f"paper{ext}").write_bytes(data)
        proj.set_status("queued", "paper stored")
        return proj

    def _acquire(self, source, content, filename) -> tuple[bytes, str]:
        if content is not None:
            return content, Path(filename).name
        s = str(source or "").strip()
        if not s:
            raise IngestError("nothing to upload: give a path, URL, DOI, arXiv id or file content")
        p = Path(s).expanduser()
        if p.is_file():
            return p.read_bytes(), p.name
        if s.startswith(("http://", "https://")) and not clean_arxiv(s):
            data, why = self.scholar.download_pdf(s)
            if data is None:
                raise IngestError(f"could not download {s}: {why}")
            return data, Path(s.split("?")[0]).name or "paper.pdf"
        arx, doi = clean_arxiv(s), clean_doi(s)
        w = self.scholar.resolve(arxiv_id=arx, doi=None if arx else doi) if (arx or doi) else None
        if w is None:
            raise IngestError(f"could not resolve {s!r} to a paper (not a file, URL, DOI or arXiv id)")
        for url in self.scholar.find_pdf_candidates(w):
            data, _ = self.scholar.download_pdf(url)
            if data:
                return data, slug(w.title) + ".pdf"
        raise IngestError(
            f"resolved to “{w.title}” but no open-access PDF was found. Get it yourself and upload the file.")

    # -- step 2..5 --------------------------------------------------------
    def process(self, proj: Project, *, depth: int | None = None, analyze: bool = True,
                progress: Progress = lambda m: None) -> Project:
        depth = self.cfg.citation_depth if depth is None else depth

        def say(msg: str) -> None:
            proj.set_status("running", msg)
            progress(msg)

        try:
            src = next(proj.primary_dir.glob("paper.*"))
            say("extracting text")
            text, pdf_meta = load_document(src)
            proj.text_path.write_text(text)
            _, ref_text = split_references(text)
            ids = guess_identifiers(text, pdf_meta)

            say("resolving paper in public databases")
            work = self.scholar.resolve(doi=ids.get("doi"), arxiv_id=ids.get("arxiv_id"), title=ids.get("title", ""))
            if work is None:
                work = Work(title=ids.get("title") or proj.id, doi=ids.get("doi"), arxiv_id=ids.get("arxiv_id"))
                progress("paper not found in public databases; continuing with text-only metadata")
            proj.save_meta({**work.to_dict(), "id": proj.id, "source_file": src.name,
                            "ref_entries_parsed": len(split_reference_entries(ref_text))})

            say("crawling citations")
            library.crawl(proj, self.scholar, work, ref_text, depth=depth,
                          max_papers=self.cfg.max_papers,
                          max_refs_per_paper=self.cfg.max_refs_per_paper, progress=say)
            library.build_missing(proj)

            if analyze:
                if self.llm is None:
                    raise RuntimeError("no LLM provider configured; run `backtrack provider add`")
                Analyzer(self.llm, proj).run(progress=say)
                report.write(proj)
                proj.set_status("done", "analysis complete")
            else:
                proj.set_status("fetched", "citations fetched; analysis skipped")
        except Exception as e:
            proj.set_status("error", f"{type(e).__name__}: {e}")
            raise
        return proj

    def reanalyze(self, proj: Project, force: list[str] | None = None,
                  progress: Progress = lambda m: None) -> None:
        if self.llm is None:
            raise RuntimeError("no LLM provider configured")
        try:
            proj.set_status("running", "re-analysing")
            Analyzer(self.llm, proj).run(force=force or [], progress=progress)
            report.write(proj)
            proj.set_status("done", "analysis complete")
        except Exception as e:
            proj.set_status("error", f"{type(e).__name__}: {e}")
            raise

    def rescan(self, proj: Project, reanalyze: bool = False, progress: Progress = lambda m: None) -> dict:
        result = library.rescan_dropbox(proj)
        if result["matched"] and reanalyze and self.llm:
            self.reanalyze(proj, force=["related", "position", "prerequisites", "learning_path"], progress=progress)
        elif result["matched"] and proj.load_stage("explain"):
            report.write(proj)
        return result


def decode_b64(data: str) -> bytes:
    try:
        return base64.b64decode(data, validate=False)
    except Exception as e:
        raise IngestError(f"invalid base64 content: {e}") from e


def safe_filename(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name)
    return name or "paper.pdf"


def build(cfg: Config, provider: str | None = None, require_llm: bool = True) -> Pipeline:
    """Wire up workspace, scholarly sources and (optionally) the configured LLM."""
    from .config import workspace_dir
    from .llm import LLMError, client_from_config

    llm = None
    try:
        llm = client_from_config(cfg, provider)
    except (KeyError, LLMError):
        if require_llm:
            raise
    scholar = Scholar(email=cfg.contact_email, s2_key=cfg.semantic_scholar_key)
    return Pipeline(cfg, Workspace(workspace_dir(cfg)), scholar, llm)
