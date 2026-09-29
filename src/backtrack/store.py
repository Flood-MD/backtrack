"""On-disk project layout.

    <workspace>/<project-id>/
        project.json      metadata about the primary paper
        status.json       pipeline progress
        primary/          paper.pdf (or original file) + paper.txt
        library/          index.json + downloaded PDFs of cited works (<key>.pdf)
        dropbox/          drop PDFs you fetched yourself for the 'missing' list here
        analysis/         one JSON file per analysis stage
        report.md         human-readable report
        missing.md/.json  works that could not be retrieved
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path

ID_RE = re.compile(r"^[a-z0-9][a-z0-9\-]{0,80}$")


class StoreError(ValueError):
    pass


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


class Project:
    def __init__(self, root: Path):
        self.root = root
        self.id = root.name

    # paths
    @property
    def primary_dir(self): return self.root / "primary"
    @property
    def library_dir(self): return self.root / "library"
    @property
    def dropbox_dir(self): return self.root / "dropbox"
    @property
    def analysis_dir(self): return self.root / "analysis"
    @property
    def text_path(self): return self.primary_dir / "paper.txt"
    @property
    def report_path(self): return self.root / "report.md"

    def ensure_dirs(self) -> None:
        for d in (self.primary_dir, self.library_dir, self.dropbox_dir, self.analysis_dir):
            d.mkdir(parents=True, exist_ok=True)

    # metadata / status
    @property
    def meta(self) -> dict:
        return read_json(self.root / "project.json", {})

    def save_meta(self, meta: dict) -> None:
        write_json(self.root / "project.json", meta)

    @property
    def status(self) -> dict:
        return read_json(self.root / "status.json", {"state": "unknown"})

    def set_status(self, state: str, message: str = "", **extra) -> None:
        write_json(self.root / "status.json",
                   {"state": state, "message": message, "updated": time.time(), **extra})

    # library index: key -> record
    def library(self) -> dict[str, dict]:
        return read_json(self.library_dir / "index.json", {})

    def save_library(self, lib: dict) -> None:
        write_json(self.library_dir / "index.json", lib)

    # analysis stages
    def stage_path(self, stage: str) -> Path:
        return self.analysis_dir / f"{stage}.json"

    def load_stage(self, stage: str):
        return read_json(self.stage_path(stage))

    def save_stage(self, stage: str, data) -> None:
        write_json(self.stage_path(stage), data)

    def paper_text(self) -> str:
        try:
            return self.text_path.read_text()
        except FileNotFoundError:
            return ""

    def summary(self) -> dict:
        m = self.meta
        lib = self.library()
        return {
            "id": self.id,
            "title": m.get("title", ""),
            "authors": m.get("authors", []),
            "year": m.get("year"),
            "doi": m.get("doi"),
            "state": self.status.get("state"),
            "message": self.status.get("message", ""),
            "related": len(lib),
            "missing": sum(1 for r in lib.values() if r.get("status") == "missing"),
            "stages": sorted(p.stem for p in self.analysis_dir.glob("*.json")) if self.analysis_dir.exists() else [],
        }


class Workspace:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, project_id: str) -> Project:
        if not ID_RE.match(project_id or ""):
            raise StoreError(f"invalid paper id {project_id!r}")
        p = self.root / project_id
        if not p.is_dir():
            raise StoreError(f"no such paper {project_id!r}")
        return Project(p)

    def create(self, base: str) -> Project:
        base = re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")[:48] or "paper"
        pid, n = base, 2
        while (self.root / pid).exists():
            pid, n = f"{base}-{n}", n + 1
        proj = Project(self.root / pid)
        proj.ensure_dirs()
        return proj

    def projects(self) -> list[Project]:
        return [Project(p) for p in sorted(self.root.iterdir())
                if p.is_dir() and (p / "project.json").exists()]

    def search(self, query: str) -> list[dict]:
        """Case-insensitive term match over titles, abstracts, TL;DRs and concept names."""
        terms = [t for t in re.split(r"\s+", query.lower()) if t]
        hits = []
        for p in self.projects():
            m = p.meta
            levels = p.load_stage("explain") or {}
            prereq = p.load_stage("prerequisites") or {}
            hay = " ".join([
                m.get("title", ""), " ".join(m.get("authors", [])), m.get("abstract", ""),
                str(levels.get("tldr", "")),
                " ".join(c.get("name", "") for c in prereq.get("concepts", [])),
            ]).lower()
            score = sum(hay.count(t) for t in terms)
            if terms and all(t in hay for t in terms):
                hits.append({**p.summary(), "score": score})
        return sorted(hits, key=lambda h: -h["score"])


def copy_into(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
