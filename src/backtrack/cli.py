"""Command line interface."""
from __future__ import annotations

import argparse
import getpass
import json
import sys

from . import __version__
from .config import PRESETS, load_config, new_provider, save_config
from .ingest import IngestError
from .llm import LLMError, make_client
from .pipeline import build
from .store import StoreError


def _say(msg: str) -> None:
    print(f"  · {msg}", file=sys.stderr)


def cmd_provider(a, cfg) -> int:
    if a.action == "add":
        key = a.api_key
        preset = PRESETS[a.preset]
        if not key and not a.api_key_env and preset["api_key_env"]:
            import os
            if not os.environ.get(preset["api_key_env"]) and sys.stdin.isatty():
                key = getpass.getpass(f"API key for {a.preset} (leave blank to use ${preset['api_key_env']}): ")
        p = new_provider(a.preset, a.name, model=a.model, api_key=key, api_key_env=a.api_key_env, base_url=a.base_url)
        if not p.model:
            print("error: --model is required for this provider", file=sys.stderr)
            return 2
        cfg.providers[p.name] = p
        if a.default or not cfg.default_provider:
            cfg.default_provider = p.name
        print(f"saved provider {p.name!r} -> {save_config(cfg)}")
    elif a.action == "list":
        for n, p in cfg.providers.items():
            star = "*" if n == cfg.default_provider else " "
            print(f"{star} {n:14} {p.kind:10} {p.model:40} key={'set' if p.resolve_key() else 'MISSING'}")
        if not cfg.providers:
            print("no providers configured; presets:", ", ".join(PRESETS))
    elif a.action == "default":
        cfg.provider(a.name)
        cfg.default_provider = a.name
        save_config(cfg)
    elif a.action == "remove":
        cfg.providers.pop(a.name, None)
        if cfg.default_provider == a.name:
            cfg.default_provider = next(iter(cfg.providers), "")
        save_config(cfg)
    elif a.action == "test":
        client = make_client(cfg.provider(a.name))
        print(client.complete("Reply with exactly: ok", "ping", max_tokens=20).strip())
    return 0


def cmd_config(a, cfg) -> int:
    fields = {"email": ("contact_email", str), "s2-key": ("semantic_scholar_key", str),
              "workspace": ("workspace", str), "depth": ("citation_depth", int),
              "max-papers": ("max_papers", int), "max-refs": ("max_refs_per_paper", int)}
    if a.key is None:
        print(json.dumps({k: getattr(cfg, v[0]) for k, v in fields.items()}, indent=2))
        return 0
    if a.key not in fields:
        print(f"unknown key; choose from {', '.join(fields)}", file=sys.stderr)
        return 2
    attr, typ = fields[a.key]
    setattr(cfg, attr, typ(a.value))
    save_config(cfg)
    return 0


def cmd_analyze(a, cfg) -> int:
    pipe = build(cfg, a.provider, require_llm=not a.no_analyze)
    proj = pipe.create_project(a.source)
    print(f"project {proj.id}", file=sys.stderr)
    pipe.process(proj, depth=a.depth, analyze=not a.no_analyze, progress=_say)
    print(f"\n{proj.root}")
    missing = json.loads((proj.root / "missing.json").read_text()) if (proj.root / "missing.json").exists() else []
    if missing:
        print(f"{len(missing)} works could not be retrieved: see {proj.root / 'missing.md'}\n"
              f"drop PDFs into {proj.dropbox_dir} and run: backtrack rescan {proj.id}")
    if not a.no_analyze:
        print(f"report: {proj.report_path}")
    return 0


def cmd_rescan(a, cfg) -> int:
    pipe = build(cfg, a.provider, require_llm=a.reanalyze)
    proj = pipe.ws.get(a.paper)
    res = pipe.rescan(proj, reanalyze=a.reanalyze, progress=_say)
    print(json.dumps(res, indent=2))
    return 0


def cmd_reanalyze(a, cfg) -> int:
    pipe = build(cfg, a.provider)
    pipe.reanalyze(pipe.ws.get(a.paper), force=a.stage or ["all"], progress=_say)
    return 0


def cmd_list(a, cfg) -> int:
    pipe = build(cfg, require_llm=False)
    for p in pipe.ws.projects():
        s = p.summary()
        print(f"{s['id']:40} {s['state']:8} missing={s['missing']:<3} {s['title'][:60]}")
    return 0


def cmd_show(a, cfg) -> int:
    proj = build(cfg, require_llm=False).ws.get(a.paper)
    if a.section == "report":
        print(proj.report_path.read_text() if proj.report_path.exists() else "(no report yet)")
    elif a.section == "missing":
        print((proj.root / "missing.md").read_text() if (proj.root / "missing.md").exists() else "(none)")
    else:
        print(json.dumps(proj.load_stage(a.section), indent=2, ensure_ascii=False))
    return 0


def cmd_ask(a, cfg) -> int:
    from .analysis import Analyzer
    pipe = build(cfg, a.provider)
    print(Analyzer(pipe.llm, pipe.ws.get(a.paper)).ask(a.question))
    return 0


def cmd_search(a, cfg) -> int:
    for h in build(cfg, require_llm=False).ws.search(" ".join(a.query)):
        print(f"{h['id']:40} {h['title'][:70]}")
    return 0


def cmd_mcp(a, cfg) -> int:
    from .mcp_server import main as mcp_main
    mcp_main(["--http"] if a.http else [])
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="backtrack", description="Total understanding of an academic paper.")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("provider", help="manage LLM providers")
    ps = p.add_subparsers(dest="action", required=True)
    add = ps.add_parser("add")
    add.add_argument("preset", choices=list(PRESETS))
    add.add_argument("--name")
    add.add_argument("--model")
    add.add_argument("--api-key")
    add.add_argument("--api-key-env")
    add.add_argument("--base-url")
    add.add_argument("--default", action="store_true")
    ps.add_parser("list")
    for n in ("default", "remove", "test"):
        x = ps.add_parser(n)
        x.add_argument("name", nargs="?" if n == "test" else None)
    p.set_defaults(fn=cmd_provider)

    c = sub.add_parser("config", help="show or set options")
    c.add_argument("key", nargs="?")
    c.add_argument("value", nargs="?")
    c.set_defaults(fn=cmd_config)

    an = sub.add_parser("analyze", help="analyse a paper (file, URL, DOI or arXiv id)")
    an.add_argument("source")
    an.add_argument("--depth", type=int, help="citation depth (default from config)")
    an.add_argument("--provider")
    an.add_argument("--no-analyze", action="store_true", help="only fetch citations; skip the LLM")
    an.set_defaults(fn=cmd_analyze)

    rs = sub.add_parser("rescan", help="pick up PDFs dropped into a project's dropbox/")
    rs.add_argument("paper")
    rs.add_argument("--reanalyze", action="store_true")
    rs.add_argument("--provider")
    rs.set_defaults(fn=cmd_rescan)

    ra = sub.add_parser("reanalyze")
    ra.add_argument("paper")
    ra.add_argument("--stage", action="append", help="stage to redo (repeatable); default all")
    ra.add_argument("--provider")
    ra.set_defaults(fn=cmd_reanalyze)

    sub.add_parser("list").set_defaults(fn=cmd_list)
    sh = sub.add_parser("show")
    sh.add_argument("paper")
    sh.add_argument("section", nargs="?", default="report",
                    choices=["report", "missing", "explain", "structure", "related", "position",
                             "prerequisites", "learning_path"])
    sh.set_defaults(fn=cmd_show)
    q = sub.add_parser("ask")
    q.add_argument("paper")
    q.add_argument("question")
    q.add_argument("--provider")
    q.set_defaults(fn=cmd_ask)
    s = sub.add_parser("search")
    s.add_argument("query", nargs="+")
    s.set_defaults(fn=cmd_search)
    m = sub.add_parser("mcp", help="run the MCP server (stdio by default)")
    m.add_argument("--http", action="store_true")
    m.set_defaults(fn=cmd_mcp)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return args.fn(args, load_config())
    except (KeyError, LLMError, IngestError, StoreError, RuntimeError) as e:
        print(f"error: {e.args[0] if isinstance(e, KeyError) else e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
