import asyncio
import base64

import pytest

from backtrack import mcp_server as m


@pytest.fixture
def server(env, monkeypatch):
    pipe, pdf, tmp = env
    monkeypatch.setattr(m, "_pipe", lambda require_llm=False: pipe)
    return pipe, pdf


def test_tools_registered():
    names = {t.name for t in asyncio.run(m.mcp.list_tools())}
    assert {"list_papers", "search_papers", "upload_paper", "get_analysis", "get_missing", "explain_paper",
            "get_learning_path", "get_prerequisite_graph", "add_missing_pdf", "rescan_paper", "ask_paper",
            "get_status", "get_paper"} <= names


def test_upload_and_query(server):
    pipe, pdf = server
    up = m.upload_paper(content_base64=base64.b64encode(pdf.read_bytes()).decode(), filename="../evil name.pdf", wait=True)
    pid = up["paper_id"]
    assert m.get_status(pid)["state"] == "done"
    assert [p["id"] for p in m.list_papers()] == [pid]
    assert m.search_papers("attention")[0]["id"] == pid and m.search_papers("zebra") == []
    assert m.explain_paper(pid, "expert") == "expert explanation"
    assert m.get_learning_path(pid, "1m")["label"] == "1 month"
    assert m.get_prerequisite_graph(pid, "mermaid").startswith("graph BT")
    assert len(m.get_missing(pid)["missing"]) == 4
    assert "# Attention Patterns" in m.get_analysis(pid, "report")
    assert m.ask_paper(pid, "why?") == "answer"
    assert m.get_paper(pid)["doi"] == "10.1234/primary.2024"


def test_path_guard_and_bad_ids(server):
    _, pdf = server
    m._allow_local_paths = False
    try:
        with pytest.raises(ValueError, match="local paths are disabled"):
            m.upload_paper(source=str(pdf))
    finally:
        m._allow_local_paths = True
    with pytest.raises(Exception):
        m.get_status("../../etc")
    with pytest.raises(Exception):
        m.get_status("nonexistent")
    with pytest.raises(ValueError):
        m.upload_paper(content_base64=base64.b64encode(b"junk").decode())


def test_add_missing_pdf(server):
    from conftest import PAYWALLED_PDF
    _, pdf = server
    pid = m.upload_paper(source=str(pdf), wait=True)["paper_id"]
    res = m.add_missing_pdf(pid, "x.pdf", content_base64=base64.b64encode(PAYWALLED_PDF).decode())
    assert res["matched"][0]["key"] == "doi-10-2222-paywall-2"
