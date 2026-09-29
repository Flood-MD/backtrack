"""Render analysis stages into report.md."""
from __future__ import annotations

from .analysis import mermaid
from .prompts import HORIZONS, LEVELS
from .store import Project


def render(project: Project) -> str:
    m = project.meta
    ex = project.load_stage("explain") or {}
    st = project.load_stage("structure") or {}
    rel = project.load_stage("related") or {}
    pos = project.load_stage("position") or {}
    pre = project.load_stage("prerequisites") or {}
    paths = project.load_stage("learning_path") or {}
    lib = project.library()
    out: list[str] = [f"# {m.get('title') or project.id}", ""]
    meta_bits = [", ".join(m.get("authors", [])[:6]), str(m.get("year") or ""), m.get("doi") and f"https://doi.org/{m['doi']}"]
    out += [" · ".join(b for b in meta_bits if b), ""]
    if ex.get("truncated_input"):
        out += ["> Note: the paper was long; the middle was omitted from the LLM input.", ""]
    if ex:
        out += ["## In brief", "", ex.get("tldr", ""), "", "**Contribution.** " + ex.get("contribution", ""), ""]
        out += ["## Explanations by level", ""]
        for k in LEVELS:
            out += [f"### {k.capitalize()}", "", ex.get("levels", {}).get(k, ""), ""]
    if st:
        out += ["## Structure and content", "", f"**Problem.** {st.get('problem', '')}", ""]
        for s in st.get("sections", []):
            out.append(f"- **{s.get('heading', '')}**: {s.get('summary', '')}")
        out += ["", "### Key claims", ""]
        out += [f"- {c.get('claim')} — *{c.get('strength', '?')}*: {c.get('evidence', '')}" for c in st.get("key_claims", [])]
        for title, key in (("Methods", "methods"), ("Results", "results"), ("Assumptions", "assumptions"),
                           ("Limitations", "limitations"), ("Open questions", "open_questions")):
            if st.get(key):
                out += ["", f"### {title}", ""] + [f"- {x}" for x in st[key]]
        if st.get("glossary"):
            out += ["", "### Glossary", ""] + [f"- **{g.get('term')}**: {g.get('definition')}" for g in st["glossary"]]
        out.append("")
    if pos:
        out += ["## What the paper is doing in its field", "",
                f"**Field:** {pos.get('field', '')} / {pos.get('subfield', '')}  ",
                f"**Primary role:** {pos.get('primary_role')}"
                + (f"  \n**Also:** {', '.join(pos['secondary_roles'])}" if pos.get("secondary_roles") else ""), "",
                pos.get("summary", ""), "",
                f"**Before this paper:** {pos.get('prevailing_view_before', '')}", "",
                f"**If it is right:** {pos.get('what_changes_if_right', '')}", ""]
        for r in pos.get("roles", []):
            tg = f" (targets: {', '.join(r['targets'])})" if r.get("targets") else ""
            out.append(f"- *{r.get('role')}*{tg}: {r.get('evidence', '')}")
        out += ["", f"_{pos.get('reception_caveat', '')}_", ""]
    if rel:
        out += ["## Surrounding research", ""]
        land = rel.get("landscape") or {}
        if land.get("narrative"):
            out += [land["narrative"], ""]
        for t in land.get("themes", []):
            out.append(f"- **{t.get('name')}**: {t.get('description')} ({', '.join(t.get('works', []))})")
        out += ["", "| Depth | Work | Relation | Note | Full text |", "|---|---|---|---|---|"]
        for e in rel.get("works", []):
            link = f"[{e['citation']}]({e['link']})" if e.get("link") else e["citation"]
            out.append(f"| {e['depth']} | {link} | {e['relation']} | {e['note']} | "
                       f"{'yes' if e['status'] == 'retrieved' else 'missing'} |")
        if land.get("coverage_note"):
            out += ["", f"_{land['coverage_note']}_"]
        out.append("")
    if pre:
        out += ["## Prerequisites", "", pre.get("summary", ""), "", "```mermaid", mermaid(pre), "```", ""]
        by_id = {c["id"]: c for c in pre["concepts"]}
        for cid in pre["order"]:
            c = by_id[cid]
            deps = f" _(needs: {', '.join(by_id[d]['name'] for d in c['depends_on'])})_" if c["depends_on"] else ""
            out.append(f"- **{c['name']}** [{c['level']}]{deps}: {c['description']}")
        out.append("")
    if paths:
        out += ["## Learning paths", ""]
        for hz in HORIZONS:
            p = paths.get(hz)
            if not p:
                continue
            out += [f"### {p['label']} (~{p['budget_hours']} h)", "", f"**Goal:** {p['goal']}  ",
                    f"**Assumes:** {p['assumes']}", ""]
            if p.get("warning"):
                out += [f"> {p['warning']}", ""]
            for i, s in enumerate(p["steps"], 1):
                out.append(f"{i}. **{s['title']}** ({s['hours']} h)")
                out += [f"   - {a}" for a in s["activities"]]
                for r in s["resources"]:
                    label = r.get("key") if r["type"] == "library" else f"{r['title']} (external, unverified)"
                    out.append(f"   - Resource: {label}. {r.get('note', '')}")
                out.append(f"   - Checkpoint: {s['checkpoint']}")
            if p.get("skipped"):
                out += ["", f"Skipped at this depth: {', '.join(p['skipped'])}"]
            out.append("")
    missing = [(k, r) for k, r in lib.items() if r["status"] in ("missing", "unresolved")]
    if missing:
        out += ["## Missing", "",
                f"{len(missing)} cited work(s) could not be retrieved; see `missing.md` and drop PDFs in `dropbox/`.", ""]
    return "\n".join(out).rstrip() + "\n"


def write(project: Project) -> None:
    project.report_path.write_text(render(project))
