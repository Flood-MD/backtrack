"""Citation crawl, PDF retrieval, the 'missing' list and dropbox rescanning."""
from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

from .ingest import (IngestError, entry_arxiv, entry_doi, extract_pdf_text, is_pdf,
                     split_reference_entries)
from .sources import Scholar, Work, norm_title, title_similarity
from .store import Project

Progress = Callable[[str], None]
EXCERPT_CHARS = 6000


def _record(work: Work, depth: int) -> dict:
    return {"work": work.to_dict(), "depth": depth, "cited_by": [], "status": "pending",
            "pdf": None, "attempts": [], "excerpt": ""}


def _raw_key(entry: str) -> str:
    return "raw-" + hashlib.sha1(entry.encode()).hexdigest()[:10]


def retrieve_pdf(scholar: Scholar, project: Project, rec: dict) -> None:
    """Try every open-access route; mark the record retrieved or missing with reasons."""
    work = Work.from_dict(rec["work"])
    dest = project.library_dir / f"{work.key}.pdf"
    if dest.exists():
        rec.update(status="retrieved", pdf=dest.name)
        return
    attempts: list[str] = []
    for url in scholar.find_pdf_candidates(work):
        data, why = scholar.download_pdf(url)
        if data is not None:
            dest.write_bytes(data)
            rec.update(status="retrieved", pdf=dest.name, attempts=attempts + [f"OK {url}"])
            rec["work"] = work.to_dict()
            _fill_excerpt(rec, dest)
            return
        attempts.append(f"{url}: {why}")
    if not attempts:
        attempts.append("no open-access PDF location found in OpenAlex, Unpaywall, Semantic Scholar or arXiv")
    rec.update(status="missing", attempts=attempts)
    rec["work"] = work.to_dict()


def _fill_excerpt(rec: dict, pdf: Path) -> None:
    try:
        text, _ = extract_pdf_text(pdf, max_pages=6)
        rec["excerpt"] = text[:EXCERPT_CHARS]
    except IngestError:
        rec["excerpt"] = ""


def _resolve_entries(scholar: Scholar, entries: list[str]) -> tuple[list[Work], list[str]]:
    found, unresolved = [], []
    for e in entries:
        doi, arx = entry_doi(e), entry_arxiv(e)
        w = scholar.resolve(doi=doi, arxiv_id=arx, citation_text=e) if (doi or arx or len(e) > 30) else None
        # Guard against Crossref's habit of returning *something*: title must appear in the entry.
        if w and not doi and not arx and title_similarity(w.title, e[: len(w.title) + 20]) < 0.6 \
                and norm_title(w.title) not in norm_title(e):
            w = None
        (found if w else unresolved).append(w or e)
    return found, unresolved  # type: ignore[return-value]


def crawl(project: Project, scholar: Scholar, primary: Work, ref_text: str, *, depth: int,
          max_papers: int, max_refs_per_paper: int, progress: Progress = lambda m: None) -> dict:
    lib = project.library()
    cap_reached = False

    def add(work: Work, d: int, parent: str) -> dict | None:
        nonlocal cap_reached
        k = work.key
        if k in lib:
            if parent not in lib[k]["cited_by"]:
                lib[k]["cited_by"].append(parent)
            return None
        if len(lib) >= max_papers:
            cap_reached = True
            return None
        rec = _record(work, d)
        rec["cited_by"] = [parent]
        lib[k] = rec
        return rec

    primary_key = "primary"
    level: list[Work] = []

    # Level 1: references of the primary paper.
    entries = split_reference_entries(ref_text)
    oa_refs = scholar.openalex_batch(primary.reference_ids) if primary.reference_ids else []
    cands = list(oa_refs)
    if len(oa_refs) < 0.6 * len(entries):
        progress(f"resolving {len(entries)} parsed reference entries")
        by_key = {w.key for w in cands}
        # Skip entries OpenAlex already gave us (title appears in the entry text).
        known = [norm_title(w.title) for w in cands if len(norm_title(w.title)) > 15]
        entries = [e for e in entries if not any(t in norm_title(e) for t in known)]
        found, unresolved = _resolve_entries(scholar, entries)
        cands += [w for w in found if w.key not in by_key]
        for e in unresolved:
            rec = _record(Work(title=e[:200]), 1)
            rec.update(status="unresolved", cited_by=[primary_key],
                       attempts=["could not match this reference string in OpenAlex/Crossref/arXiv"])
            rec["raw"] = e
            lib.setdefault(_raw_key(e), rec)
    for w in cands:
        if add(w, 1, primary_key):
            level.append(w)

    for d in range(1, max(depth, 1) + 1):
        # Include leftovers from an interrupted earlier run.
        todo = {w.key: w for w in level}
        for k, r in lib.items():
            if r["depth"] == d and r["status"] == "pending":
                todo.setdefault(k, Work.from_dict(r["work"]))
        for i, (k, w) in enumerate(todo.items(), 1):
            progress(f"depth {d}: fetching {i}/{len(todo)} {w.title[:60]}")
            retrieve_pdf(scholar, project, lib[k])
            project.save_library(lib)
        if d == depth or not level:
            break
        nxt: list[Work] = []
        for w in level:
            if not w.reference_ids:
                continue
            refs = sorted(scholar.openalex_batch(w.reference_ids), key=lambda r: -r.cited_by_count)
            for r in refs[:max_refs_per_paper]:
                if add(r, d + 1, w.key):
                    nxt.append(r)
        level = nxt
    if cap_reached:
        progress(f"stopped adding papers at the max_papers cap ({max_papers})")
    project.save_library(lib)
    return lib


def build_missing(project: Project) -> list[dict]:
    lib = project.library()
    titles = {k: r["work"].get("title") or r.get("raw", "")[:80] for k, r in lib.items()}
    items = []
    for k, r in lib.items():
        if r["status"] not in ("missing", "unresolved"):
            continue
        w = Work.from_dict(r["work"])
        items.append({
            "key": k,
            "citation": r.get("raw") or w.citation(),
            "link": w.link(),
            "doi": w.doi,
            "depth": r["depth"],
            "cited_by": [titles.get(c, "primary paper") if c != "primary" else "the primary paper"
                         for c in r["cited_by"]],
            "reason": r["status"],
            "tried": r["attempts"],
            "save_as": f"{k}.pdf",
        })
    items.sort(key=lambda i: (i["depth"], -len(i["cited_by"]), i["citation"]))
    from .store import write_json
    write_json(project.root / "missing.json", items)
    lines = [
        f"# Missing works for `{project.id}`", "",
        "These could not be retrieved from public sources. Get them through your library or "
        "institutional access and drop the PDFs into `dropbox/` (any filename works), then run "
        f"`backtrack rescan {project.id}`.", "",
    ]
    if not items:
        lines.append("Nothing is missing.")
    for i in items:
        lines.append(f"- [ ] **{i['citation']}**")
        if i["link"]:
            lines.append(f"  - Link: {i['link']}")
        lines.append(f"  - Depth {i['depth']}; cited by: {'; '.join(i['cited_by'])[:200]}")
        if i["reason"] == "unresolved":
            lines.append("  - Could not be identified automatically; search for it by hand.")
        else:
            lines.append(f"  - Tried: {len(i['tried'])} source(s) — {i['tried'][0][:160] if i['tried'] else ''}")
    (project.root / "missing.md").write_text("\n".join(lines) + "\n")
    return items


def rescan_dropbox(project: Project) -> dict:
    """Match PDFs the user dropped in dropbox/ to missing works and move them into the library."""
    lib = project.library()
    matched, unmatched = [], []
    for pdf in sorted(project.dropbox_dir.glob("*")):
        if not pdf.is_file():
            continue
        if pdf.suffix.lower() != ".pdf" or not is_pdf(pdf.read_bytes()[:16]):
            unmatched.append({"file": pdf.name, "reason": "not a PDF"})
            continue
        key = _match_dropped(pdf, lib)
        if not key:
            unmatched.append({"file": pdf.name, "reason": "did not match any missing work"})
            continue
        dest = project.library_dir / f"{key}.pdf"
        pdf.replace(dest)
        rec = lib[key]
        rec.update(status="retrieved", pdf=dest.name)
        rec["attempts"].append("supplied manually via dropbox")
        _fill_excerpt(rec, dest)
        matched.append({"file": pdf.name, "key": key})
    project.save_library(lib)
    build_missing(project)
    return {"matched": matched, "unmatched": unmatched}


def _match_dropped(pdf: Path, lib: dict) -> str | None:
    missing = {k: r for k, r in lib.items() if r["status"] in ("missing", "unresolved")}
    if pdf.stem in missing:
        return pdf.stem
    try:
        text, _ = extract_pdf_text(pdf, max_pages=2)
    except IngestError:
        return None
    head, head_n = text[:4000], norm_title(text[:4000])
    for k, r in missing.items():
        w = Work.from_dict(r["work"])
        if w.doi and w.doi in head.lower():
            return k
        t = norm_title(w.title)
        if len(t) > 15 and t in head_n:
            return k
    return None
