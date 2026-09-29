import json

from backtrack.analysis import clean_graph
from backtrack.sources import Work, clean_arxiv, clean_doi
from conftest import PAYWALLED_PDF, make_pdf


def test_identifier_parsing():
    assert clean_doi("see https://doi.org/10.1000/ABC.def-1).") == "10.1000/abc.def-1"
    assert clean_arxiv("https://arxiv.org/abs/1706.03762v5") == "1706.03762"
    assert clean_arxiv("arXiv:1706.03762") == "1706.03762"
    assert clean_arxiv("hello") is None


def test_work_merge_and_key():
    a = Work(title="T", doi="10.1/x")
    a.merge(Work(title="other", abstract="abs", pdf_urls=["u"], cited_by_count=5))
    assert (a.title, a.abstract, a.pdf_urls, a.cited_by_count, a.key) == ("T", "abs", ["u"], 5, "doi-10-1-x")


def test_clean_graph_breaks_cycles_and_orders():
    g = clean_graph([
        {"id": "a", "name": "A", "depends_on": ["c", "a", "zzz"]},
        {"id": "b", "name": "B", "depends_on": ["a"]},
        {"id": "c", "name": "C", "depends_on": ["b"], "level": "bogus", "library_works": ["k1", "k2"]},
    ], valid_lib={"k1"})
    assert g["removed_cycle_edges"] and set(g["order"]) == {"a", "b", "c"}
    pos = {n: i for i, n in enumerate(g["order"])}
    for c in g["concepts"]:
        assert all(pos[d] < pos[c["id"]] for d in c["depends_on"])
    c = next(c for c in g["concepts"] if c["id"] == "c")
    assert c["level"] == "intermediate" and c["library_works"] == ["k1"]


def test_full_pipeline(env):
    pipe, pdf, tmp = env
    proj = pipe.create_project(pdf)
    pipe.process(proj, depth=2)

    assert proj.status["state"] == "done"
    assert proj.meta["doi"] == "10.1234/primary.2024" and proj.meta["ref_entries_parsed"] == 4
    lib = proj.library()
    by_status = {}
    for k, r in lib.items():
        by_status.setdefault(r["status"], []).append(k)
    assert by_status["retrieved"] == ["doi-10-1111-widgets-1"]
    assert (proj.library_dir / "doi-10-1111-widgets-1.pdf").exists()
    # paywalled paper and depth-2 grandparent are missing; the unidentifiable entry is unresolved
    assert set(by_status["missing"]) == {"doi-10-2222-paywall-2", "doi-10-3333-gp-4"}
    assert lib["doi-10-3333-gp-4"]["depth"] == 2 and lib["doi-10-3333-gp-4"]["cited_by"] == ["doi-10-1111-widgets-1"]
    assert len(by_status["unresolved"]) == 2

    missing = json.loads((proj.root / "missing.json").read_text())
    assert {m["reason"] for m in missing} == {"missing", "unresolved"}
    assert any("not a PDF" in t for m in missing for t in m["tried"])
    assert "dropbox/" in (proj.root / "missing.md").read_text()

    # LLM output is validated against the library
    rel = proj.load_stage("related")
    assert {w["key"] for w in rel["works"]} == set(by_status["retrieved"] + by_status["missing"])
    assert "bogus" not in rel["landscape"]["themes"][0]["works"]
    assert rel["unresolved_references"] == 2
    pos = proj.load_stage("position")
    assert pos["secondary_roles"] == [] and "bogus" not in pos["roles"][0]["targets"]
    pre = proj.load_stage("prerequisites")
    assert pre["removed_cycle_edges"] and pre["concepts"][0]["library_works"] == []
    paths = proj.load_stage("learning_path")
    assert set(paths) == {"1h", "1w", "1m", "1y"}
    step = paths["1h"]["steps"][0]
    assert step["concepts"] == ["attention"] and step["resources"][0]["unverified"] is True

    text = proj.report_path.read_text()
    for needle in ("## Explanations by level", "```mermaid", "## Learning paths", "## Missing", "Small transformers"):
        assert needle in text

    # cached stages are not recomputed
    n = len(pipe.llm.calls)
    pipe.reanalyze(proj)
    assert len(pipe.llm.calls) == n

    # user supplies the paywalled PDF, under an arbitrary filename
    (proj.dropbox_dir / "scan_0001.pdf").write_bytes(PAYWALLED_PDF)
    (proj.dropbox_dir / "junk.pdf").write_bytes(make_pdf(["unrelated"] * 5))
    (proj.dropbox_dir / "notes.txt").write_text("x")
    res = pipe.rescan(proj)
    assert res["matched"] == [{"file": "scan_0001.pdf", "key": "doi-10-2222-paywall-2"}]
    assert {u["file"] for u in res["unmatched"]} == {"junk.pdf", "notes.txt"}
    assert proj.library()["doi-10-2222-paywall-2"]["status"] == "retrieved"
    assert "doi-10-2222-paywall-2" not in {m["key"] for m in json.loads((proj.root / "missing.json").read_text())}


def test_fetch_only_without_llm(env):
    pipe, pdf, _ = env
    pipe.llm = None
    proj = pipe.create_project(pdf)
    pipe.process(proj, depth=1, analyze=False)
    assert proj.status["state"] == "fetched" and (proj.root / "missing.md").exists()
    assert proj.load_stage("explain") is None


def test_error_is_recorded(env):
    pipe, pdf, _ = env
    pipe.llm = None
    proj = pipe.create_project(pdf)
    try:
        pipe.process(proj)
    except RuntimeError:
        pass
    assert proj.status["state"] == "error" and "provider" in proj.status["message"]


def test_bad_uploads(env):
    import pytest
    from backtrack.ingest import IngestError
    pipe, _, tmp = env
    (tmp / "fake.pdf").write_bytes(b"not a pdf")
    with pytest.raises(IngestError):
        pipe.create_project(tmp / "fake.pdf")
    with pytest.raises(IngestError):
        pipe.create_project("10.9999/unknown")
    with pytest.raises(IngestError):
        pipe.create_project("")
