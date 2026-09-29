"""Prompt templates. Kept in one place so they are easy to tune."""

SYSTEM = (
    "You are a meticulous research analyst helping a reader reach total understanding of an "
    "academic work in any discipline. Be accurate and specific; ground every claim in the "
    "supplied text. If the text does not support something, say so rather than guessing. "
    "Never invent citations, numbers, or results."
)

LEVELS = {
    "layperson": "an intelligent adult with no background in the field; no jargon, use analogies",
    "undergraduate": "a student who has taken introductory courses in the broad discipline",
    "graduate": "a graduate student in an adjacent subfield; may use standard technical terms",
    "expert": "a specialist in this exact subfield; focus on what is novel, technical detail, and caveats",
}

ROLES = {
    "proposing": "introduces a new method, theory, dataset, system or result",
    "extending": "builds directly on prior work to go further",
    "modifying": "changes an existing method or theory to improve or adapt it",
    "challenging": "argues against a prevailing view or prior finding",
    "critiquing": "analyses flaws or limits in existing work or methods",
    "reviewing": "surveys and organises existing literature",
    "synthesizing": "unifies disparate strands into a common framework",
    "replicating": "tests whether an earlier result holds again",
    "applying": "uses established methods on a new problem or domain",
    "theorizing": "develops conceptual or formal theory",
    "commentary": "opinion, position or perspective piece",
}

EXPLAIN = """Explain the paper below at four levels of expertise.

Return JSON:
{{
  "tldr": "one or two sentences",
  "contribution": "what is new here, in one paragraph",
  "levels": {{
{level_keys}
  }}
}}

Audience for each level:
{level_desc}

Each level value is Markdown, 200-500 words, and must be self-contained. The layperson level
should define everything it uses; the expert level should be dense and specific.

PAPER (title: {title}):
{text}
"""

STRUCTURE = """Analyse the structure and content of the paper below.

Return JSON:
{{
  "problem": "the problem or question addressed and why it matters",
  "sections": [{{"heading": "...", "summary": "2-4 sentences on what this part does"}}],
  "key_claims": [{{"claim": "...", "evidence": "how the paper supports it", "strength": "strong|moderate|weak|unclear"}}],
  "methods": ["..."],
  "results": ["specific findings, with numbers where the paper gives them"],
  "assumptions": ["..."],
  "limitations": ["stated by the authors, or evident from the text (label which)"],
  "open_questions": ["..."],
  "glossary": [{{"term": "...", "definition": "..."}}]
}}

PAPER (title: {title}):
{text}
"""

RELATED_BATCH = """The primary paper is "{title}". Abstract/summary:
{summary}

For each cited work below, judge how the primary paper relates to it, using only the
information given. Returns JSON:
{{"works": [{{"key": "<key exactly as given>",
             "relation": "builds_on|extends|challenges|critiques|uses_method|uses_data|background|compares_against|unclear",
             "note": "one sentence: what the cited work contributes and how the primary paper uses it"}}]}}

CITED WORKS:
{works}
"""

LANDSCAPE = """The primary paper is "{title}". Abstract/summary:
{summary}

Here are works it cites (and, at greater depth, works those cite), with the relation to the
primary paper already assessed:
{works}

Write a map of the surrounding research. Returns JSON:
{{
  "narrative": "3-6 sentence Markdown story of the research lineage leading to this paper",
  "themes": [{{"name": "...", "description": "...", "works": ["<key>", ...]}}],
  "timeline": [{{"year": 0, "key": "<key>", "why_it_matters": "..."}}],
  "coverage_note": "what this map cannot see because works were unavailable or the crawl was capped"
}}
Use only keys from the list above.
"""

POSITION = """Decide what role this paper plays in its field.

Roles you may assign (choose one primary and any number of secondary):
{roles}

Return JSON:
{{
  "field": "...", "subfield": "...",
  "primary_role": "<role>",
  "secondary_roles": ["<role>", ...],
  "summary": "one paragraph: what the paper is doing in its field and against/with whom",
  "roles": [{{"role": "<role>", "evidence": "quote or specific pointer from the paper",
             "targets": ["<library key of the work it acts on>", ...]}}],
  "prevailing_view_before": "what the field believed or did before",
  "what_changes_if_right": "what changes if the paper is right",
  "reception_caveat": "note that this is inferred from the text and cited works, not from later citations"
}}
Targets must be keys from the library list or empty.

PAPER (title: {title}):
{text}

ANALYSIS OF RELATED WORK:
{related}
"""

PREREQS = """List what a reader must understand to fully grasp this paper, as a dependency graph.

Return JSON:
{{
  "summary": "one paragraph on the shape of the prerequisite landscape",
  "concepts": [
    {{"id": "short-kebab-id",
      "name": "...",
      "description": "what to know, 1-3 sentences",
      "level": "foundational|intermediate|advanced",
      "why_needed": "how the paper depends on it",
      "depends_on": ["<id of another concept in this list>"],
      "library_works": ["<library key of a cited work that teaches or introduces it>"],
      "study_hint": "type of resource to look for (textbook chapter, seminal paper, tutorial)"}}
  ]
}}

Rules: 12-40 concepts; every depends_on id must be defined; no cycles; foundational
concepts have empty depends_on; the paper's own core ideas are the top of the graph
(include one concept with id "paper-core" that depends on the concepts it directly needs).
library_works keys must come from the library list, else leave it empty.

PAPER (title: {title}):
{text}

LIBRARY (cited works):
{library}
"""

PATH = """Design a learning path that takes a reader from their starting point to understanding
the paper "{title}" within a fixed time budget.

Horizon: {horizon} (about {hours} hours of focused study in total).
{level_hint}

Prerequisite concepts in a valid study order (id: name [level] - depends_on):
{concepts}

Cited works available to the user (key: citation, retrieved yes/no):
{library}

Return JSON:
{{
  "goal": "what the reader will be able to do at the end",
  "assumes": "assumed starting knowledge",
  "skipped": ["<concept id deliberately left out at this depth>"],
  "steps": [{{"title": "...", "concepts": ["<concept id>"], "hours": 0.0,
             "activities": ["concrete tasks: read, derive, code, explain aloud..."],
             "resources": [{{"type": "library|external", "key": "<library key if library>",
                            "title": "for external: a well-known textbook/course/paper", "note": "..."}}],
             "checkpoint": "how the reader can tell they got it"}}]
}}

The step hours must sum to about {hours}. Cover the concepts that matter most first and
say honestly what is skipped. Only reference library keys from the list; label anything else
as external and only name resources you are confident exist.
"""

HORIZONS = {
    "1h": {"label": "1 hour", "hours": 1,
           "hint": "Give the shortest route to a working grasp: only essential ideas, no side trips."},
    "1w": {"label": "1 week", "hours": 10,
           "hint": "Assume about 1.5 hours a day. Cover the essential and intermediate concepts."},
    "1m": {"label": "1 month", "hours": 40,
           "hint": "Assume about 10 hours a week. Include exercises and reading the key cited works."},
    "1y": {"label": "1 year", "hours": 250,
           "hint": "Assume about 5 hours a week. Build the foundations properly, then work toward "
                   "the research frontier; include a project or reproduction."},
}

ASK = """Answer the question about the paper using only the material below. If the material does
not contain the answer, say so. Be concise and point to the relevant section.

QUESTION: {question}

PAPER (title: {title}):
{text}

ANALYSIS NOTES:
{notes}
"""
