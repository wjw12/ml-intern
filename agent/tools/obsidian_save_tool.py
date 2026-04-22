"""Obsidian vault save tool — write new paper notes into the research vault.

Handles metadata fetch, PDF download, full-text extraction, slug generation,
citation stamping, and index update. The calling agent provides the note body
(TL;DR through Key Links); this tool wraps it in proper frontmatter.

Replicates logic from /srv/syncthing/obsidian-lab/agents/compile.py.
"""

from __future__ import annotations

import os
import re
import time
from datetime import date
from pathlib import Path
from typing import Any

import httpx

VAULT_ROOT = Path(os.environ.get("OBSIDIAN_VAULT_PATH", "/srv/syncthing/obsidian-lab/Lab"))
WIKI_PAPERS = VAULT_ROOT / "wiki" / "papers"
INDEX_FILE = VAULT_ROOT / "wiki" / "_index.md"
ATTACHMENTS = VAULT_ROOT / "attachments"
CLIPPINGS = VAULT_ROOT / "Clippings"

S2_API = "https://api.semanticscholar.org/graph/v1/paper"


def _make_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:80]


def _fetch_arxiv_metadata(arxiv_id: str) -> dict:
    base_id = re.sub(r"v\d+$", "", arxiv_id)
    url = f"https://export.arxiv.org/api/query?id_list={base_id}"
    meta: dict[str, Any] = {"title": "", "authors": [], "year": "", "first_author_lastname": ""}
    for attempt in range(4):
        try:
            r = httpx.get(url, timeout=15)
            if r.status_code == 429:
                time.sleep(5 * (2 ** attempt))
                continue
            r.raise_for_status()
            xml = r.text
            titles = re.findall(r"<title>(.+?)</title>", xml, re.DOTALL)
            if len(titles) >= 2:
                meta["title"] = re.sub(r"\s+", " ", titles[1]).strip()
            authors = re.findall(r"<name>(.+?)</name>", xml)
            meta["authors"] = [a.strip().split()[-1] for a in authors]
            meta["first_author_lastname"] = meta["authors"][0] if meta["authors"] else ""
            year_match = re.search(r"<published>(\d{4})", xml)
            if year_match:
                meta["year"] = year_match.group(1)
            break
        except Exception:
            break
    return meta


def _fetch_citation_count(arxiv_id: str) -> tuple[int, int]:
    url = f"{S2_API}/ArXiv:{arxiv_id}?fields=citationCount,influentialCitationCount"
    for attempt in range(3):
        try:
            r = httpx.get(url, timeout=15)
            if r.status_code == 429:
                time.sleep(5 * (2 ** attempt))
                continue
            if r.status_code == 404:
                return 0, 0
            r.raise_for_status()
            d = r.json()
            return d.get("citationCount", 0), d.get("influentialCitationCount", 0)
        except Exception:
            return 0, 0
    return 0, 0


def _download_pdf(arxiv_id: str) -> Path | None:
    base_id = re.sub(r"v\d+$", "", arxiv_id)
    pdf_path = ATTACHMENTS / f"{base_id}.pdf"
    if pdf_path.exists():
        return pdf_path
    try:
        r = httpx.get(
            f"https://arxiv.org/pdf/{base_id}",
            follow_redirects=True, timeout=60,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        r.raise_for_status()
        ATTACHMENTS.mkdir(exist_ok=True)
        pdf_path.write_bytes(r.content)
        return pdf_path
    except Exception:
        return None


def _extract_fulltext(arxiv_id: str, pdf_path: Path) -> Path | None:
    id_slug = arxiv_id.replace(".", "-")
    full_md = CLIPPINGS / f"{id_slug}-full.md"
    if full_md.exists():
        return full_md
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        pages = [page.get_text() for page in doc]
        doc.close()
        CLIPPINGS.mkdir(exist_ok=True)
        full_md.write_text(
            f"<!-- full text extracted from arxiv:{arxiv_id} -->\n\n" + "\n\n".join(pages),
            encoding="utf-8",
        )
        return full_md
    except ImportError:
        return None


def _resolve_slug(arxiv_id: str, meta: dict) -> str:
    parts = [p for p in [meta.get("first_author_lastname", ""), meta.get("year", "")] if p]
    prefix = _make_slug("-".join(parts))
    title_slug = _make_slug(meta.get("title", arxiv_id))
    return f"{prefix}-{title_slug}" if prefix else title_slug


def _extract_tldr(text: str) -> str:
    m = re.search(r"^##\s+TL;DR\s*\n+(.+?)(?:\n\s*\n|\n##\s)", text, re.DOTALL | re.MULTILINE)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


def _append_to_index(slug: str, tldr: str) -> None:
    if not INDEX_FILE.exists():
        return
    text = INDEX_FILE.read_text(encoding="utf-8")
    if f"[[{slug}]]" in text:
        return
    entry = f"[[{slug}]] — {tldr}"
    if "## Papers" in text:
        text = text.replace("## Papers\n", f"## Papers\n{entry}\n", 1)
    else:
        text += f"\n{entry}\n"
    INDEX_FILE.write_text(text, encoding="utf-8")


async def obsidian_save_handler(arguments: dict[str, Any]) -> tuple[str, bool]:
    arxiv_id = arguments.get("arxiv_id", "").strip()
    content = arguments.get("content", "").strip()

    if not arxiv_id:
        return "'arxiv_id' is required.", False
    if not content:
        return "'content' is required — provide the note body (TL;DR through Key Links).", False

    meta = _fetch_arxiv_metadata(arxiv_id)
    if not meta["title"]:
        return f"Could not fetch metadata for arxiv:{arxiv_id}.", False

    slug = _resolve_slug(arxiv_id, meta)
    note_path = WIKI_PAPERS / f"{slug}.md"

    if note_path.exists():
        return f"Note already exists: {note_path.name}", False

    pdf_path = _download_pdf(arxiv_id)
    if pdf_path:
        _extract_fulltext(arxiv_id, pdf_path)

    citations, influential = _fetch_citation_count(arxiv_id)
    today = date.today().isoformat()
    base_id = re.sub(r"v\d+$", "", arxiv_id)
    authors_str = ", ".join(meta["authors"][:10])
    pdf_embed = f"\n![[{base_id}.pdf]]" if pdf_path else ""

    # Build frontmatter + wrap content
    note = (
        f"---\n"
        f'title: "{meta["title"]}"\n'
        f"authors: [{authors_str}]\n"
        f"year: {meta['year']}\n"
        f'arxiv: "{arxiv_id}"\n'
        f'venue: ""\n'
        f"tags: []\n"
        f"added: {today}\n"
        f"citations: {citations}\n"
        f"influential_citations: {influential}\n"
        f"---\n"
        f"{pdf_embed}\n\n"
        f"{content}\n"
    )

    WIKI_PAPERS.mkdir(parents=True, exist_ok=True)
    note_path.write_text(note, encoding="utf-8")

    tldr = _extract_tldr(note)
    _append_to_index(slug, tldr or meta["title"])

    return (
        f"Saved: wiki/papers/{slug}.md\n"
        f"Title: {meta['title']}\n"
        f"Authors: {authors_str}\n"
        f"Citations: {citations} ({influential} influential)\n"
        f"PDF: {'downloaded' if pdf_path else 'failed'}\n"
        f"Index: updated"
    ), True


OBSIDIAN_SAVE_TOOL_SPEC = {
    "name": "obsidian_save",
    "description": (
        "Save a new paper note to the Obsidian research vault.\n\n"
        "Provide an arxiv_id and the note body content (## TL;DR through ## Key Links). "
        "The tool auto-generates frontmatter from arxiv metadata, downloads the PDF, "
        "stamps citation counts, and updates the vault index.\n\n"
        "Do NOT include YAML frontmatter in content — it is generated automatically.\n\n"
        "Use hf_papers(operation='read_paper') to get paper content for writing the summary. "
        "Use obsidian_kb(operation='graph') on related vault papers to write Key Links."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "arxiv_id": {
                "type": "string",
                "description": "ArXiv paper ID (e.g. '2412.06769').",
            },
            "content": {
                "type": "string",
                "description": (
                    "Note body markdown starting from ## TL;DR through ## Key Links. "
                    "Must include sections: TL;DR, Key Contributions, Method, Results, "
                    "Limitations, Key Links. Do NOT include frontmatter."
                ),
            },
        },
        "required": ["arxiv_id", "content"],
    },
}
