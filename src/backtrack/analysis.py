"""LLM analysis stages. Each stage caches its result as analysis/<stage>.json."""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable

from . import prompts as P
from .ingest import excerpt_for_llm, split_references
from .llm import LLMClient
from .sources import Work, slug
from .store import Project

STAGES = ["explain", "structure", "related", "position", "prerequisites", "learning_path"]
Progress = Callable[[str], None]


def _library_view(project: Project, include_unretrieved: bool = True) -> dict[str, dict]:
    out = {}
    for k, r in project.library().items():
        if r["status"] == "unresolved":
            continue
        if not include_unretrieved and r["status"] != "retrieved":
            continue
        out[k] = r
    return out


def _work_line(key: str, r: dict, abstract_chars: int = 500) -> str:
    w = Work.from_dict(r["work"])
    info = w.abstract or r.get("excerpt", "")
    info = re.sub(r"\s+", " ", info)[:abstract_chars] or "(no abstract available)"
    have = "full text read" if r["status"] == "retrieved" else "metadata only"
    return f"- key={key} | depth {r['depth']} | {w.citation()} | {have}\n  {info}"


def clean_graph(concepts: list[dict], valid_lib: Iterable[str] = ()) -> dict:
    """Normalise ids, drop dangling/self edges, break cycles, and topologically sort."""
    valid_lib = set(valid_lib)
    by_id: dict[str, dict] = {}
    for c in concepts:
        cid = slug(str(c.get("id") or c.get("name") or ""), 60)
        if not c.get("name") or cid in by_id:
            continue
        by_id[cid] = {
            "id": cid, "name": c["name"], "description": c.get("description", ""),
            "level": c.get("level") if c.get("level") in ("foundational", "intermediate", "advanced") else "intermediate",
            "why_needed": c.get("why_needed", ""), "study_hint": c.get("study_hint", ""),
            "depends_on": [slug(str(d), 60) for d in c.get("depends_on") or []],
            "library_works": [k for k in c.get("library_works") or [] if k in valid_lib],
        }
    for c in by_id.values():
        c["depends_on"] = list(dict.fromkeys(d for d in c["depends_on"] if d in by_id and d != c["id"]))
    # Break cycles: DFS, drop back-edges.
    state: dict[str, int] = {}
    removed: list[tuple[str, str]] = []

    def visit(n: str) -> None:
        state[n] = 1
        for d in list(by_id[n]["depends_on"]):
            if state.get(d) == 1:
                by_id[n]["depends_on"].remove(d)
                removed.append((n, d))
            elif d not in state:
                visit(d)
        state[n] = 2

    for n in list(by_id):
        if n not in state:
            visit(n)
    # Kahn's algorithm: prerequisites first.
    indeg = {n: len(c["depends_on"]) for n, c in by_id.items()}
    order, ready = [], sorted(n for n, d in indeg.items() if d == 0)
    while ready:
        n = ready.pop(0)
        order.append(n)
        for m, c in by_id.items():
            if n in c["depends_on"]:
                indeg[m] -= 1
                if indeg[m] == 0:
                    ready.append(m)
    return {"concepts": list(by_id.values()), "order": order,
            "removed_cycle_edges": [{"from": a, "to": b} for a, b in removed]}


def mermaid(graph: dict) -> str:
    lines = ["graph BT"]
    shape = {"foundational": "([{}])", "intermediate": "[{}]", "advanced": "{{{{{}}}}}"}
    for c in graph["concepts"]:
        label = c["name"].replace('"', "'")
        lines.append(f'  {c["id"].replace("-", "_")}' + shape[c["level"]].format(f'"{label}"'))
    for c in graph["concepts"]:
        for d in c["depends_on"]:
            lines.append(f'  {d.replace("-", "_")} --> {c["id"].replace("-", "_")}')
    return "\n".join(lines)


class Analyzer:
    def __init__(self, llm: LLMClient, project: Project, max_chars: int = 90_000):
        self.llm, self.project, self.max_chars = llm, project, max_chars
        self.meta = project.meta
        body, _ = split_references(project.paper_text())
        self.text, self.truncated = excerpt_for_llm(body, max_chars)
        self.title = self.meta.get("title") or project.id
        self.summary = self.meta.get("abstract") or self.text[:1500]

    def _ask_json(self, prompt: str, max_tokens: int = 6000):
        return self.llm.complete_json(P.SYSTEM, prompt, max_tokens=max_tokens)

    def run(self, force: Iterable[str] = (), progress: Progress = lambda m: None) -> dict:
        force = set(force)
        if "all" in force:
            force = set(STAGES)
        results = {}
        for stage in STAGES:
            cached = self.project.load_stage(stage)
            if cached is not None and stage not in force:
                results[stage] = cached
                continue
            progress(f"analysis: {stage}")
            results[stage] = getattr(self, f"stage_{stage}")()
            self.project.save_stage(stage, results[stage])
        return results

    # -- stages -----------------------------------------------------------
    def stage_explain(self) -> dict:
        prompt = P.EXPLAIN.format(
            level_keys="\n".join(f'    "{k}": "markdown",' for k in P.LEVELS).rstrip(","),
            level_desc="\n".join(f"- {k}: {v}" for k, v in P.LEVELS.items()),
            title=self.title, text=self.text,
        )
        data = self._ask_json(prompt, 8000)
        levels = data.get("levels") or {}
        data["levels"] = {k: str(levels.get(k, "")) for k in P.LEVELS}
        data["truncated_input"] = self.truncated
        return data

    def stage_structure(self) -> dict:
        return self._ask_json(P.STRUCTURE.format(title=self.title, text=self.text), 8000)

    def stage_related(self) -> dict:
        lib = _library_view(self.project)
        ordered = sorted(lib.items(), key=lambda kv: (kv[1]["depth"], -kv[1]["work"].get("cited_by_count", 0)))
        works: dict[str, dict] = {}
        for i in range(0, len(ordered), 8):
            batch = ordered[i:i + 8]
            data = self._ask_json(P.RELATED_BATCH.format(
                title=self.title, summary=self.summary[:1500],
                works="\n".join(_work_line(k, r) for k, r in batch)), 3000)
            for item in data.get("works", []):
                if item.get("key") in lib:
                    works[item["key"]] = {"relation": item.get("relation", "unclear"), "note": item.get("note", "")}
        entries = []
        for k, r in ordered:
            w = Work.from_dict(r["work"])
            entries.append({"key": k, "citation": w.citation(), "link": w.link(), "year": w.year,
                            "depth": r["depth"], "status": r["status"],
                            **works.get(k, {"relation": "unclear", "note": ""})})
        landscape = {}
        if entries:
            listing = "\n".join(f"- key={e['key']} ({e['year']}) {e['citation']} [{e['relation']}] {e['note']}"
                                for e in entries[:80])
            landscape = self._ask_json(P.LANDSCAPE.format(
                title=self.title, summary=self.summary[:1500], works=listing), 4000)
            keys = {e["key"] for e in entries}
            for t in landscape.get("themes", []):
                t["works"] = [k for k in t.get("works", []) if k in keys]
            landscape["timeline"] = [t for t in landscape.get("timeline", []) if t.get("key") in keys]
        unresolved = sum(1 for r in self.project.library().values() if r["status"] == "unresolved")
        return {"works": entries, "landscape": landscape, "unresolved_references": unresolved}

    def stage_position(self) -> dict:
        related = self.project.load_stage("related") or {}
        rel_txt = "\n".join(f"- key={e['key']} [{e['relation']}] {e['citation']}: {e['note']}"
                            for e in related.get("works", [])[:60]) or "(no related works retrieved)"
        data = self._ask_json(P.POSITION.format(
            roles="\n".join(f"- {k}: {v}" for k, v in P.ROLES.items()),
            title=self.title, text=self.text, related=rel_txt), 4000)
        keys = {e["key"] for e in related.get("works", [])}
        for r in data.get("roles", []):
            r["targets"] = [t for t in r.get("targets", []) if t in keys]
        if data.get("primary_role") not in P.ROLES:
            data["primary_role"] = next((r["role"] for r in data.get("roles", []) if r.get("role") in P.ROLES),
                                        "commentary")
        data["secondary_roles"] = [r for r in data.get("secondary_roles", []) if r in P.ROLES]
        return data

    def stage_prerequisites(self) -> dict:
        lib = _library_view(self.project)
        listing = "\n".join(_work_line(k, r, 200) for k, r in lib.items()) or "(none)"
        data = self._ask_json(P.PREREQS.format(title=self.title, text=self.text, library=listing), 8000)
        graph = clean_graph(data.get("concepts", []), lib.keys())
        graph["summary"] = data.get("summary", "")
        return graph

    def stage_learning_path(self) -> dict:
        graph = self.project.load_stage("prerequisites") or {"concepts": [], "order": []}
        by_id = {c["id"]: c for c in graph["concepts"]}
        concepts = "\n".join(
            f"{i}: {by_id[i]['name']} [{by_id[i]['level']}] - {', '.join(by_id[i]['depends_on']) or 'none'}"
            for i in graph["order"])
        lib = _library_view(self.project)
        libtxt = "\n".join(f"{k}: {Work.from_dict(r['work']).citation()} "
                           f"({'retrieved' if r['status'] == 'retrieved' else 'not retrieved'})"
                           for k, r in lib.items()) or "(none)"
        paths = {}
        for hz, spec in P.HORIZONS.items():
            data = self._ask_json(P.PATH.format(
                title=self.title, horizon=spec["label"], hours=spec["hours"], level_hint=spec["hint"],
                concepts=concepts or "(none)", library=libtxt), 6000)
            paths[hz] = self._clean_path(hz, spec, data, by_id, lib)
        return paths

    @staticmethod
    def _clean_path(hz: str, spec: dict, data: dict, by_id: dict, lib: dict) -> dict:
        steps = []
        for s in data.get("steps", []):
            res = []
            for r in s.get("resources", []):
                if r.get("type") == "library" and r.get("key") in lib:
                    res.append({"type": "library", "key": r["key"], "note": r.get("note", "")})
                else:
                    res.append({"type": "external", "title": r.get("title") or r.get("key", ""),
                                "note": r.get("note", ""), "unverified": True})
            try:
                hours = float(s.get("hours", 0))
            except (TypeError, ValueError):
                hours = 0.0
            steps.append({"title": s.get("title", ""), "hours": hours,
                          "concepts": [c for c in s.get("concepts", []) if c in by_id],
                          "activities": s.get("activities", []), "resources": res,
                          "checkpoint": s.get("checkpoint", "")})
        total = round(sum(s["hours"] for s in steps), 2)
        out = {"horizon": hz, "label": spec["label"], "budget_hours": spec["hours"], "total_hours": total,
               "goal": data.get("goal", ""), "assumes": data.get("assumes", ""),
               "skipped": [c for c in data.get("skipped", []) if c in by_id], "steps": steps}
        if total and abs(total - spec["hours"]) > 0.25 * spec["hours"]:
            out["warning"] = f"step hours sum to {total}, budget is {spec['hours']}"
        return out

    # -- Q&A --------------------------------------------------------------
    def ask(self, question: str) -> str:
        notes = {s: self.project.load_stage(s) for s in ("explain", "structure", "position")}
        blob = json.dumps({k: v for k, v in notes.items() if v}, ensure_ascii=False)[:20_000]
        return self.llm.complete(P.SYSTEM, P.ASK.format(question=question, title=self.title,
                                                        text=self.text, notes=blob), max_tokens=2000)
