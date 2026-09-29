"""Public academic database connectors: OpenAlex, Crossref, Semantic Scholar, arXiv, Unpaywall.

Every lookup is best-effort: network or API failures are recorded in ``Scholar.errors``
and never raised, so one flaky service cannot sink a crawl.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher

import httpx

DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+)", re.I)
ARXIV_RE = re.compile(r"(?:arxiv[:\s/]*)?\b(\d{4}\.\d{4,5})(?:v\d+)?\b|\b([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?\b", re.I)
_TRAILING = ".,;)]}>'\""


def clean_doi(s: str) -> str | None:
    m = DOI_RE.search(s or "")
    return m.group(1).rstrip(_TRAILING).lower() if m else None


def clean_arxiv(s: str) -> str | None:
    s = (s or "").strip()
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([^\s?#]+?)(?:\.pdf)?(?:v\d+)?$", s, re.I)
    if m:
        return m.group(1)
    m = re.fullmatch(r"(?:arxiv:)?(\d{4}\.\d{4,5})(?:v\d+)?", s, re.I)
    if m:
        return m.group(1)
    m = re.fullmatch(r"(?:arxiv:)?([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?", s, re.I)
    return m.group(1) if m else None


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", re.sub(r"\s+", " ", (t or "").lower())).strip()


def title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, norm_title(a), norm_title(b)).ratio()


def slug(s: str, n: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:n].strip("-") or "paper"


@dataclass
class Work:
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    openalex_id: str | None = None  # e.g. "W2741809807"
    abstract: str = ""
    venue: str = ""
    cited_by_count: int = 0
    pdf_urls: list[str] = field(default_factory=list)
    reference_ids: list[str] = field(default_factory=list)  # OpenAlex ids of cited works
    reference_dois: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        """Stable, filesystem-safe id, used for PDF filenames and the library index."""
        if self.doi:
            return "doi-" + re.sub(r"[^a-z0-9]+", "-", self.doi.lower()).strip("-")
        if self.arxiv_id:
            return "arxiv-" + re.sub(r"[^a-z0-9]+", "-", self.arxiv_id.lower()).strip("-")
        if self.openalex_id:
            return "oa-" + self.openalex_id.lower()
        return "t-" + slug(self.title, 60)

    def citation(self) -> str:
        who = ", ".join(self.authors[:3]) + (" et al." if len(self.authors) > 3 else "")
        bits = [b for b in (who, f"({self.year})" if self.year else "", self.title, self.venue) if b]
        return " ".join(bits)

    def link(self) -> str:
        if self.doi:
            return f"https://doi.org/{self.doi}"
        if self.arxiv_id:
            return f"https://arxiv.org/abs/{self.arxiv_id}"
        if self.openalex_id:
            return f"https://openalex.org/{self.openalex_id}"
        return ""

    def merge(self, other: "Work") -> "Work":
        """Fill gaps in self from other (self wins on conflicts)."""
        for f in ("title", "year", "doi", "arxiv_id", "openalex_id", "abstract", "venue"):
            if not getattr(self, f) and getattr(other, f):
                setattr(self, f, getattr(other, f))
        if not self.authors:
            self.authors = other.authors
        self.cited_by_count = max(self.cited_by_count, other.cited_by_count)
        for f in ("pdf_urls", "reference_ids", "reference_dois", "sources"):
            mine = getattr(self, f)
            mine.extend(x for x in getattr(other, f) if x not in mine)
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Work":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _uninvert(idx: dict | None) -> str:
    if not idx:
        return ""
    words: dict[int, str] = {}
    for w, positions in idx.items():
        for p in positions:
            words[p] = w
    return " ".join(words[i] for i in sorted(words))


class Scholar:
    """Aggregates the public databases. Pass an ``httpx.Client`` to inject a transport in tests."""

    def __init__(self, email: str = "", s2_key: str = "", http: httpx.Client | None = None):
        self.email = email
        self.s2_key = s2_key
        ua = f"backtrack/0.1 (mailto:{email})" if email else "backtrack/0.1"
        self.http = http or httpx.Client(timeout=30, follow_redirects=True, headers={"User-Agent": ua})
        self.errors: list[str] = []

    # -- plumbing ---------------------------------------------------------
    def _get(self, url: str, params: dict | None = None, headers: dict | None = None) -> httpx.Response | None:
        try:
            r = self.http.get(url, params=params, headers=headers)
        except httpx.HTTPError as e:
            self.errors.append(f"{url}: {e}")
            return None
        if r.status_code == 200:
            return r
        if r.status_code != 404:
            self.errors.append(f"{url}: HTTP {r.status_code}")
        return None

    def _json(self, url, params=None, headers=None):
        r = self._get(url, params, headers)
        if r is None:
            return None
        try:
            return r.json()
        except ValueError:
            return None

    # -- OpenAlex ---------------------------------------------------------
    def _oa_params(self) -> dict:
        return {"mailto": self.email} if self.email else {}

    @staticmethod
    def _oa_work(d: dict) -> Work:
        loc_pdfs = [
            loc["pdf_url"] for loc in (d.get("locations") or []) if loc.get("pdf_url")
        ]
        oa_url = (d.get("open_access") or {}).get("oa_url")
        pdfs = list(dict.fromkeys(loc_pdfs + ([oa_url] if oa_url and oa_url.lower().endswith(".pdf") else [])))
        doi = clean_doi(d.get("doi") or "")
        arxiv = None
        for loc in d.get("locations") or []:
            arxiv = arxiv or clean_arxiv(loc.get("landing_page_url") or "")
        return Work(
            title=d.get("title") or d.get("display_name") or "",
            authors=[a["author"]["display_name"] for a in d.get("authorships") or [] if a.get("author")],
            year=d.get("publication_year"),
            doi=doi,
            arxiv_id=arxiv,
            openalex_id=(d.get("id") or "").rsplit("/", 1)[-1] or None,
            abstract=_uninvert(d.get("abstract_inverted_index")),
            venue=((d.get("primary_location") or {}).get("source") or {}).get("display_name") or "",
            cited_by_count=d.get("cited_by_count") or 0,
            pdf_urls=pdfs,
            reference_ids=[u.rsplit("/", 1)[-1] for u in d.get("referenced_works") or []],
            sources=["openalex"],
        )

    def openalex_get(self, *, doi: str | None = None, oa_id: str | None = None) -> Work | None:
        ident = f"https://doi.org/{doi}" if doi else oa_id
        d = self._json(f"https://api.openalex.org/works/{ident}", self._oa_params())
        return self._oa_work(d) if isinstance(d, dict) and d.get("id") else None

    def openalex_search(self, title: str) -> list[Work]:
        d = self._json("https://api.openalex.org/works", {**self._oa_params(), "search": title, "per-page": 5})
        return [self._oa_work(x) for x in (d or {}).get("results", [])]

    def openalex_batch(self, ids: list[str]) -> list[Work]:
        out: list[Work] = []
        for i in range(0, len(ids), 50):
            chunk = ids[i : i + 50]
            d = self._json(
                "https://api.openalex.org/works",
                {**self._oa_params(), "filter": "openalex:" + "|".join(chunk), "per-page": 50},
            )
            out.extend(self._oa_work(x) for x in (d or {}).get("results", []))
        return out

    # -- Crossref ---------------------------------------------------------
    @staticmethod
    def _cr_work(m: dict) -> Work:
        year = None
        for k in ("issued", "published-print", "published-online", "created"):
            parts = (m.get(k) or {}).get("date-parts") or [[None]]
            if parts[0] and parts[0][0]:
                year = parts[0][0]
                break
        abstract = re.sub(r"<[^>]+>", " ", m.get("abstract") or "").strip()
        return Work(
            title=(m.get("title") or [""])[0],
            authors=[" ".join(filter(None, [a.get("given"), a.get("family")])) for a in m.get("author") or []],
            year=year,
            doi=clean_doi(m.get("DOI") or ""),
            abstract=re.sub(r"\s+", " ", abstract),
            venue=(m.get("container-title") or [""])[0],
            cited_by_count=m.get("is-referenced-by-count") or 0,
            pdf_urls=[
                l["URL"] for l in m.get("link") or [] if "pdf" in (l.get("content-type") or "") and l.get("URL")
            ],
            reference_dois=[clean_doi(r["DOI"]) for r in m.get("reference") or [] if r.get("DOI")],
            sources=["crossref"],
        )

    def crossref_get(self, doi: str) -> Work | None:
        d = self._json(f"https://api.crossref.org/works/{doi}", {"mailto": self.email} if self.email else None)
        return self._cr_work(d["message"]) if isinstance(d, dict) and d.get("message") else None

    def crossref_search(self, text: str, rows: int = 3) -> list[Work]:
        params = {"query.bibliographic": text, "rows": rows}
        if self.email:
            params["mailto"] = self.email
        d = self._json("https://api.crossref.org/works", params)
        return [self._cr_work(x) for x in ((d or {}).get("message") or {}).get("items", [])]

    # -- Semantic Scholar -------------------------------------------------
    def s2_get(self, ident: str) -> Work | None:
        fields = "title,year,abstract,venue,authors,externalIds,openAccessPdf,citationCount"
        h = {"x-api-key": self.s2_key} if self.s2_key else None
        d = self._json(f"https://api.semanticscholar.org/graph/v1/paper/{ident}", {"fields": fields}, h)
        if not isinstance(d, dict) or not d.get("title"):
            return None
        ext = d.get("externalIds") or {}
        pdf = (d.get("openAccessPdf") or {}).get("url")
        return Work(
            title=d["title"],
            authors=[a.get("name", "") for a in d.get("authors") or []],
            year=d.get("year"),
            doi=clean_doi(ext.get("DOI") or ""),
            arxiv_id=ext.get("ArXiv"),
            abstract=d.get("abstract") or "",
            venue=d.get("venue") or "",
            cited_by_count=d.get("citationCount") or 0,
            pdf_urls=[pdf] if pdf else [],
            sources=["semanticscholar"],
        )

    # -- arXiv ------------------------------------------------------------
    def arxiv_get(self, arxiv_id: str) -> Work | None:
        r = self._get("https://export.arxiv.org/api/query", {"id_list": arxiv_id})
        if r is None:
            return None
        return self._parse_arxiv(r.text)

    def arxiv_search_title(self, title: str) -> Work | None:
        q = 'ti:"' + re.sub(r'["\\]', " ", title) + '"'
        r = self._get("https://export.arxiv.org/api/query", {"search_query": q, "max_results": 1})
        w = self._parse_arxiv(r.text) if r is not None else None
        return w if w and title_similarity(w.title, title) >= 0.9 else None

    @staticmethod
    def _parse_arxiv(xml: str) -> Work | None:
        ns = {"a": "http://www.w3.org/2005/Atom"}
        try:
            entry = ET.fromstring(xml).find("a:entry", ns)
        except ET.ParseError:
            return None
        if entry is None or entry.find("a:title", ns) is None:
            return None
        title = re.sub(r"\s+", " ", entry.findtext("a:title", "", ns)).strip()
        if title.lower() == "error":
            return None
        aid = clean_arxiv(entry.findtext("a:id", "", ns))
        doi_el = entry.find("{http://arxiv.org/schemas/atom}doi")
        published = entry.findtext("a:published", "", ns)
        return Work(
            title=title,
            authors=[a.findtext("a:name", "", ns) for a in entry.findall("a:author", ns)],
            year=int(published[:4]) if published[:4].isdigit() else None,
            doi=clean_doi(doi_el.text) if doi_el is not None and doi_el.text else None,
            arxiv_id=aid,
            abstract=re.sub(r"\s+", " ", entry.findtext("a:summary", "", ns)).strip(),
            venue="arXiv",
            pdf_urls=[f"https://arxiv.org/pdf/{aid}"] if aid else [],
            sources=["arxiv"],
        )

    # -- Unpaywall --------------------------------------------------------
    def unpaywall_pdfs(self, doi: str) -> list[str]:
        if not self.email:
            return []
        d = self._json(f"https://api.unpaywall.org/v2/{doi}", {"email": self.email})
        if not isinstance(d, dict):
            return []
        locs = [d.get("best_oa_location")] + list(d.get("oa_locations") or [])
        return list(dict.fromkeys(l["url_for_pdf"] for l in locs if l and l.get("url_for_pdf")))

    # -- high level -------------------------------------------------------
    def resolve(self, *, doi: str | None = None, arxiv_id: str | None = None, title: str = "",
                citation_text: str = "") -> Work | None:
        """Resolve an identifier or fuzzy title/citation string to a merged Work, or None."""
        parts: list[Work | None] = []
        if arxiv_id and not doi:
            a = self.arxiv_get(arxiv_id)
            parts.append(a)
            doi = a.doi if a else None
            title = title or (a.title if a else "")
        if doi:
            parts += [self.openalex_get(doi=doi), self.crossref_get(doi)]
        elif title or citation_text:
            probe = title or citation_text
            hit = self._best_match(self.openalex_search(probe[:300]), probe)
            if hit is None:
                hit = self._best_match(self.crossref_search(citation_text or title), probe)
            if hit is None and title:
                hit = self.arxiv_search_title(title)
            if hit is None:
                return None
            parts.append(hit)
            if hit.doi:
                parts += [self.openalex_get(doi=hit.doi) if "openalex" not in hit.sources else None,
                          self.crossref_get(hit.doi) if "crossref" not in hit.sources else None]
        works = [w for w in parts if w]
        if not works:
            return None
        merged = works[0]
        for w in works[1:]:
            merged.merge(w)
        if not merged.abstract or not merged.pdf_urls:
            ident = f"DOI:{merged.doi}" if merged.doi else (f"ARXIV:{merged.arxiv_id}" if merged.arxiv_id else "")
            if ident and (s2 := self.s2_get(ident)):
                merged.merge(s2)
        return merged

    @staticmethod
    def _best_match(cands: list[Work], probe: str, threshold: float = 0.8) -> Work | None:
        best, score = None, 0.0
        pn = norm_title(probe)
        for c in cands:
            if not c.title:
                continue
            s = title_similarity(c.title, probe)
            # A raw citation string contains the title: accept containment too.
            if len(norm_title(c.title)) > 15 and norm_title(c.title) in pn:
                s = max(s, 0.95)
            if s > score:
                best, score = c, s
        return best if score >= threshold else None

    def find_pdf_candidates(self, w: Work) -> list[str]:
        urls: list[str] = []
        if w.arxiv_id:
            urls.append(f"https://arxiv.org/pdf/{w.arxiv_id}")
        elif w.title and (a := self.arxiv_search_title(w.title)):
            w.arxiv_id = a.arxiv_id
            urls += a.pdf_urls
        urls += w.pdf_urls
        if w.doi:
            urls += self.unpaywall_pdfs(w.doi)
            if s2 := self.s2_get(f"DOI:{w.doi}"):
                urls += s2.pdf_urls
        return list(dict.fromkeys(urls))

    def download_pdf(self, url: str) -> tuple[bytes | None, str]:
        """Return (bytes, "") on success or (None, reason)."""
        try:
            r = self.http.get(url, headers={"Accept": "application/pdf"})
        except httpx.HTTPError as e:
            return None, f"network error: {e}"
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}"
        if not r.content.lstrip()[:5].startswith(b"%PDF"):
            return None, "response was not a PDF (paywall or landing page)"
        if len(r.content) > 80 * 1024 * 1024:
            return None, "PDF larger than 80 MB"
        return r.content, ""
