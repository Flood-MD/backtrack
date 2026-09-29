# backtrack

Total understanding of an academic paper, book chapter or article, in any subject.
Give it a paper; it produces:

- **Explanations at four levels** (layperson, undergraduate, graduate, expert) plus a structural breakdown: problem, sections, key claims with evidence strength, methods, results, assumptions, limitations, glossary.
- **The surrounding research**: it follows citations to a depth you choose (`N`), retrieves what public databases allow, and says how the paper relates to each cited work.
- **The paper's role in its field**: proposing, extending, modifying, challenging, critiquing, reviewing, synthesizing, replicating, applying, theorizing, commentary. Each role comes with evidence and the cited works it acts on.
- **Prerequisites as a dependency graph** (Mermaid diagram plus a topologically ordered list), linked to cited works that teach each concept.
- **Learning paths for 1 hour, 1 week, 1 month and 1 year**, each with a goal, steps, hour budgets, activities, resources and checkpoints.
- **A `missing` list** of cited works it could not get from public sources, so you can fetch them through your own access and drop them in the project's `dropbox/` folder.

It also runs as an **MCP server**, so an agent can query analysed papers or submit new ones.

## Install

```bash
pip install -e .            # Python 3.10+
pip install -e '.[dev]'     # adds pytest and reportlab for the tests
```

## Set up an LLM provider

Keys can be stored in `~/.backtrack/config.json` (mode 0600) or read from an environment variable.
An environment variable always wins over a stored key.

```bash
backtrack provider add openai       --model gpt-4o                    # $OPENAI_API_KEY
backtrack provider add openrouter   --model anthropic/claude-sonnet-4.5   # $OPENROUTER_API_KEY
backtrack provider add huggingface  --model meta-llama/Llama-3.3-70B-Instruct   # $HF_TOKEN
backtrack provider add anthropic    --model claude-sonnet-4-5         # $ANTHROPIC_API_KEY
backtrack provider add custom --name ollama --base-url http://localhost:11434/v1 --model llama3.1
backtrack provider list
backtrack provider test
backtrack provider default openrouter
```

Without `--api-key`, `add` prompts for a key on a terminal (blank = use the environment variable).
The four named presets fill in the endpoint. `custom` is any OpenAI-compatible server (Ollama, vLLM, LM Studio).
Model names change often; pass `--model` for whatever you want to use.

Optional but recommended: a contact email, which OpenAlex/Crossref use for their polite pool and **Unpaywall requires** to find open-access PDFs.

```bash
backtrack config email you@university.edu
backtrack config depth 2          # default citation depth
backtrack config max-papers 40    # cap on total related works fetched
backtrack config max-refs 25      # cap on references followed per paper beyond depth 1
backtrack config s2-key <key>     # optional Semantic Scholar API key
```

## Analyse a paper

```bash
backtrack analyze paper.pdf                # a local file (.pdf, .txt, .md)
backtrack analyze 10.1038/nature14539      # a DOI (needs an open-access copy)
backtrack analyze arxiv:1706.03762         # an arXiv id or URL
backtrack analyze paper.pdf --depth 2      # follow citations two levels out
backtrack analyze paper.pdf --no-analyze   # fetch citations only, no LLM
```

Each paper becomes a project folder under `~/.backtrack/workspace/` (override with `BACKTRACK_WORKSPACE` or `backtrack config workspace`):

```
<project>/
  primary/          your paper + extracted text
  library/          index.json + PDFs of cited works
  dropbox/          <- drop PDFs you fetched yourself here
  analysis/         explain / structure / related / position / prerequisites / learning_path (.json)
  report.md         everything, readable
  missing.md/.json  what could not be retrieved, with links and what was tried
```

### The missing list

When a cited work has no open-access PDF (or the download turned out to be a paywall page), it is recorded with its DOI/link and the sources that were tried. References that could not be identified at all are listed as unresolved. To fill gaps:

```bash
backtrack show <project> missing
# fetch PDFs via your library, drop them in <project>/dropbox/ under any filename
backtrack rescan <project>                # matches by filename, DOI or title, updates the list
backtrack rescan <project> --reanalyze    # also redo the analyses that depend on related work
```

### Other commands

```bash
backtrack list
backtrack show <project> [report|missing|explain|structure|related|position|prerequisites|learning_path]
backtrack ask <project> "Why does the paper drop the second term?"
backtrack search attention
backtrack reanalyze <project> --stage prerequisites --stage learning_path
```

Analysis stages are cached in `analysis/`, so an interrupted run resumes where it stopped.

## MCP server

```bash
backtrack mcp            # stdio, for local agents
backtrack mcp --http     # streamable HTTP; refuses to read local file paths
```

Example Claude Desktop / Claude Code configuration:

```json
{ "mcpServers": { "backtrack": { "command": "backtrack-mcp" } } }
```

| Tool | Purpose |
|---|---|
| `list_papers`, `search_papers` | browse and search analysed papers |
| `get_paper`, `get_status` | metadata, TL;DR, processing state |
| `get_analysis(paper_id, section)` | `overview`, `structure`, `related`, `position`, `prerequisites`, `learning_path`, `missing`, `report` |
| `explain_paper(paper_id, level)` | explanation at a chosen level |
| `get_learning_path(paper_id, horizon)` | `1h`, `1w`, `1m`, `1y` |
| `get_prerequisite_graph(paper_id, format)` | JSON or Mermaid |
| `ask_paper(paper_id, question)` | question answering grounded in the paper |
| `upload_paper(source \| content_base64, depth, analyze, wait)` | submit a paper (local path in stdio mode, URL, DOI, arXiv id, or base64 bytes); runs in the background |
| `get_missing`, `add_missing_pdf`, `rescan_paper` | drive the missing-list workflow |

Resources: `backtrack://papers`, `backtrack://paper/{id}/report`.
Uploading and analysis use the default provider from your config, so set one up first.

## Data sources

OpenAlex (metadata, reference lists, open-access locations), Crossref, Semantic Scholar, arXiv and Unpaywall. All are queried best-effort; a failing service is skipped and its error does not stop the crawl.
Only openly available PDFs are downloaded. Downloads must actually be PDFs; paywall landing pages are rejected and land in the missing list.

## Limits worth knowing

- **Long papers**: text beyond ~90k characters is trimmed to its head and tail for the LLM (flagged in the report).
- **Scanned PDFs** have no text layer; OCR them first (e.g. `ocrmypdf`).
- **Cited works without full text** are analysed from their abstracts only, and the report says which.
- **The paper's role in its field** is inferred from the paper and its references, not from later citations.
- **Reference parsing** of the paper's own bibliography is heuristic. OpenAlex's reference list is preferred; parsed entries fill gaps.
- **External resources** in learning paths (textbooks, courses) come from the model and are marked unverified. Cited works are validated against the library.
- Tests use recorded fakes for HTTP and the LLM; the live databases and LLM APIs were not exercised in CI.

## Development

```bash
pytest
```
