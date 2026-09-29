import io
import json
import re

import httpx
import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from backtrack.config import Config
from backtrack.llm import LLMClient
from backtrack.pipeline import Pipeline
from backtrack.sources import Scholar
from backtrack.store import Workspace


def make_pdf(lines: list[str]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    y = 750
    for line in lines:
        c.drawString(40, y, line[:110])
        y -= 16
        if y < 60:
            c.showPage()
            y = 750
    c.save()
    return buf.getvalue()


PRIMARY_LINES = (
    ["Attention Patterns in Tiny Transformers", "Ada Example, Bob Sample", "doi:10.1234/primary.2024", ""]
    + [f"Body sentence {i}: we study how small transformer models allocate attention over tokens." for i in range(40)]
    + ["", "References", "[1] Vaswani A. et al. Deep Widgets for Sequence Modelling. 2017. doi:10.1111/widgets.1",
       "[2] Smith J. A Hidden Survey Nobody Can Download Freely. Journal of Paywalls, 2019.",
       "[3] Zzz Q. Unfindable manuscript about nothing at all in particular, private communication, 2020.",
       "[4] Yyy R. Another unfindable manuscript about nothing whatsoever either, private communication, 2021."]
)
WIDGET_PDF = make_pdf(["Deep Widgets for Sequence Modelling", "Vaswani et al.", "doi 10.1111/widgets.1"]
                      + ["Widgets are all you need. " * 4] * 20)
PAYWALLED_PDF = make_pdf(["A Hidden Survey Nobody Can Download Freely", "Smith", "doi 10.2222/paywall.2"]
                         + ["Survey text about paywalls. " * 4] * 20)


def oa_work(wid, title, doi, refs=(), pdf=None, year=2017, cites=100):
    return {
        "id": f"https://openalex.org/{wid}", "title": title, "doi": f"https://doi.org/{doi}" if doi else None,
        "publication_year": year, "cited_by_count": cites,
        "authorships": [{"author": {"display_name": "A. Author"}}],
        "abstract_inverted_index": {"An": [0], "abstract": [1], "about": [2], title.split()[0]: [3]},
        "referenced_works": [f"https://openalex.org/{r}" for r in refs],
        "locations": [{"pdf_url": pdf, "landing_page_url": None}] if pdf else [],
        "open_access": {"oa_url": pdf},
        "primary_location": {"source": {"display_name": "Journal of Tests"}},
    }


WORKS = {
    "10.1234/primary.2024": oa_work("W1", "Attention Patterns in Tiny Transformers", "10.1234/primary.2024",
                                    refs=["W2", "W3"], year=2024, cites=3),
    "10.1111/widgets.1": oa_work("W2", "Deep Widgets for Sequence Modelling", "10.1111/widgets.1",
                                 refs=["W4"], pdf="https://files.example/widgets.pdf"),
    "10.2222/paywall.2": oa_work("W3", "A Hidden Survey Nobody Can Download Freely", "10.2222/paywall.2",
                                 year=2019, cites=50, pdf="https://publisher.example/paywall.pdf"),
    "W4": oa_work("W4", "Grandparent Foundations of Widgetry", "10.3333/gp.4", year=1999, cites=9000),
}


def handler(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    if url.startswith("https://api.openalex.org/works/"):
        ident = url.split("/works/", 1)[1].split("?")[0].replace("https://doi.org/", "").replace("https%3A//doi.org/", "")
        w = WORKS.get(ident)
        return httpx.Response(200, json=w) if w else httpx.Response(404)
    if url.startswith("https://api.openalex.org/works"):
        flt = request.url.params.get("filter", "")
        if flt.startswith("openalex:"):
            ids = flt[len("openalex:"):].split("|")
            return httpx.Response(200, json={"results": [w for w in WORKS.values()
                                                         if w["id"].rsplit("/", 1)[-1] in ids]})
        return httpx.Response(200, json={"results": []})
    if url.startswith("https://files.example/widgets.pdf"):
        return httpx.Response(200, content=WIDGET_PDF)
    if "paywall" in url:
        return httpx.Response(200, content=b"<html>please log in</html>")
    # Everything else (crossref, s2, arxiv, unpaywall) knows nothing.
    return httpx.Response(404)


class FakeLLM(LLMClient):
    name, model = "fake", "fake-1"

    def __init__(self):
        self.calls: list[str] = []

    def complete(self, system, user, max_tokens=4096, temperature=0.2):
        self.calls.append(user[:40])
        if "Explain the paper below" in user:
            return json.dumps({"tldr": "Small transformers attend locally.", "contribution": "A study.",
                               "levels": {k: f"{k} explanation" for k in ("layperson", "undergraduate", "graduate", "expert")}})
        if "Analyse the structure" in user:
            return json.dumps({"problem": "p", "sections": [{"heading": "Intro", "summary": "s"}],
                               "key_claims": [{"claim": "c", "evidence": "e", "strength": "moderate"}],
                               "methods": ["m"], "results": ["r"], "assumptions": [], "limitations": ["l"],
                               "open_questions": [], "glossary": [{"term": "attention", "definition": "d"}]})
        if "judge how the primary paper relates" in user:
            keys = re.findall(r"key=(\S+)", user)
            return "```json\n" + json.dumps({"works": [{"key": k, "relation": "builds_on", "note": "n"} for k in keys]
                                             + [{"key": "hallucinated", "relation": "x", "note": "x"}]}) + "\n```"
        if "Write a map of the surrounding research" in user:
            keys = re.findall(r"key=(\S+)", user)
            return json.dumps({"narrative": "story", "themes": [{"name": "t", "description": "d", "works": keys + ["bogus"]}],
                               "timeline": [], "coverage_note": "c"})
        if "Decide what role" in user:
            keys = re.findall(r"key=(\S+)", user)
            return json.dumps({"field": "ML", "subfield": "NLP", "primary_role": "extending", "secondary_roles": ["nonsense"],
                               "summary": "s", "roles": [{"role": "extending", "evidence": "e", "targets": keys[:1] + ["bogus"]}],
                               "prevailing_view_before": "b", "what_changes_if_right": "w", "reception_caveat": "c"})
        if "dependency graph" in user:
            return json.dumps({"summary": "s", "concepts": [
                {"id": "linear-algebra", "name": "Linear algebra", "description": "d", "level": "foundational",
                 "depends_on": ["paper-core"]},  # cycle on purpose
                {"id": "attention", "name": "Attention", "description": "d", "level": "intermediate",
                 "depends_on": ["linear-algebra", "ghost"], "library_works": ["nope"]},
                {"id": "paper-core", "name": "Core", "description": "d", "level": "advanced", "depends_on": ["attention"]}]})
        if "Design a learning path" in user:
            return json.dumps({"goal": "g", "assumes": "a", "skipped": ["ghost"], "steps": [
                {"title": "Step", "concepts": ["attention", "ghost"], "hours": 1.0, "activities": ["read"],
                 "resources": [{"type": "library", "key": "missing-key", "title": "T"}], "checkpoint": "c"}]})
        return "answer"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKTRACK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("BACKTRACK_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setattr("time.sleep", lambda s: None)
    http = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    scholar = Scholar(email="t@example.org", http=http)
    llm = FakeLLM()
    cfg = Config(citation_depth=2, max_papers=20, max_refs_per_paper=5)
    pipe = Pipeline(cfg, Workspace(tmp_path / "ws"), scholar, llm)
    pdf = tmp_path / "primary.pdf"
    pdf.write_bytes(make_pdf(PRIMARY_LINES))
    return pipe, pdf, tmp_path
