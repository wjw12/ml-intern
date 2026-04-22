"""Obsidian knowledge-base tool — read-only access to the local research vault.

Provides search, read, list, and graph operations over compiled wiki paper
notes in an Obsidian vault.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

VAULT_ROOT = Path(os.environ.get("OBSIDIAN_VAULT_PATH", "/srv/syncthing/obsidian-lab/Lab"))
WIKI_PAPERS = VAULT_ROOT / "wiki" / "papers"
INDEX_FILE = VAULT_ROOT / "wiki" / "_index.md"

# Regex for YAML frontmatter and wikilinks
_FM_RE = re.compile(r"^---\n(.*?)\n---", re.DOTALL)
_LINK_RE = re.compile(r"\[\[([^\]|]+?)(?:\|[^\]]*)?\]\]")
_FM_FIELD = re.compile(r"^(\w[\w\s]*):\s*(.+)$", re.MULTILINE)
_FM_LIST = re.compile(r"^\s*-\s*(.+)$", re.MULTILINE)


def _parse_frontmatter(text: str) -> dict[str, str]:
    m = _FM_RE.match(text)
    if not m:
        return {}
    block = m.group(1)
    fm: dict[str, str] = {}
    for field in _FM_FIELD.finditer(block):
        key, val = field.group(1).strip(), field.group(2).strip()
        if val.startswith("["):
            items = _FM_LIST.findall(block[field.start():])
            fm[key] = ", ".join(items) if items else val.strip("[]")
        else:
            fm[key] = val.strip('"').strip("'")
    return fm


def _load_index() -> list[tuple[str, str]]:
    """Return [(slug, tl_dr_line), ...] from _index.md."""
    if not INDEX_FILE.exists():
        return []
    entries: list[tuple[str, str]] = []
    for line in INDEX_FILE.read_text().splitlines():
        m = re.match(r"\[\[([^\]]+)\]\]\s*—\s*(.+)", line)
        if m:
            entries.append((m.group(1), m.group(2)))
    return entries


def _note_path(slug: str) -> Path | None:
    p = WIKI_PAPERS / f"{slug}.md"
    return p if p.exists() else None


def _extract_links(text: str) -> list[str]:
    """Extract outgoing wikilink slugs from Key Links section."""
    in_links = False
    links: list[str] = []
    for line in text.splitlines():
        if line.strip().startswith("## Key Links"):
            in_links = True
            continue
        if in_links and line.strip().startswith("## "):
            break
        if in_links:
            for m in _LINK_RE.finditer(line):
                slug = m.group(1)
                if not slug.endswith(".pdf"):
                    links.append(slug)
    return links


# ── Operations ──────────────────────────────────────────────────────


async def _op_search(args: dict[str, Any], limit: int) -> dict:
    query = args.get("query", "").lower()
    if not query:
        return {"formatted": "'query' is required for search.", "isError": True}

    terms = query.split()
    index = _load_index()
    scored: list[tuple[int, str, str]] = []
    for slug, tldr in index:
        text = f"{slug} {tldr}".lower()
        score = sum(1 for t in terms if t in text)
        if score > 0:
            scored.append((score, slug, tldr))

    scored.sort(key=lambda x: -x[0])
    results = scored[:limit]
    if not results:
        return {"formatted": f"No papers matching '{query}' in vault."}

    lines = [f"Found {len(results)} papers:\n"]
    for score, slug, tldr in results:
        fm = {}
        p = _note_path(slug)
        if p:
            fm = _parse_frontmatter(p.read_text())
        arxiv = fm.get("arxiv", "")
        arxiv_str = f" (arxiv:{arxiv})" if arxiv else ""
        lines.append(f"- **{slug}**{arxiv_str}\n  {tldr}")
    return {"formatted": "\n".join(lines)}


async def _op_read_note(args: dict[str, Any], _limit: int) -> dict:
    slug = args.get("slug", "")
    if not slug:
        return {"formatted": "'slug' is required for read_note.", "isError": True}

    # Fuzzy match: allow partial slug
    p = _note_path(slug)
    if not p:
        candidates = list(WIKI_PAPERS.glob(f"*{slug}*.md"))
        if len(candidates) == 1:
            p = candidates[0]
        elif candidates:
            names = [c.stem for c in candidates[:10]]
            return {"formatted": f"Ambiguous slug. Matches: {', '.join(names)}", "isError": True}
        else:
            return {"formatted": f"Note '{slug}' not found in wiki/papers/.", "isError": True}

    return {"formatted": p.read_text()}


async def _op_list_papers(_args: dict[str, Any], limit: int) -> dict:
    index = _load_index()
    if not index:
        return {"formatted": "No papers in vault index."}

    entries = index[:limit]
    lines = [f"Vault contains {len(index)} papers:\n"]
    for slug, tldr in entries:
        lines.append(f"- [[{slug}]] — {tldr}")
    if len(index) > limit:
        lines.append(f"\n... and {len(index) - limit} more. Use search to narrow.")
    return {"formatted": "\n".join(lines)}


async def _op_graph(args: dict[str, Any], _limit: int) -> dict:
    slug = args.get("slug", "")
    if not slug:
        return {"formatted": "'slug' is required for graph.", "isError": True}

    p = _note_path(slug)
    if not p:
        candidates = list(WIKI_PAPERS.glob(f"*{slug}*.md"))
        if len(candidates) == 1:
            p = candidates[0]
        else:
            return {"formatted": f"Note '{slug}' not found.", "isError": True}

    text = p.read_text()
    slug_actual = p.stem

    # Outgoing links (from Key Links section)
    outgoing = _extract_links(text)

    # Incoming links (other notes that link to this one)
    incoming: list[str] = []
    for other in WIKI_PAPERS.glob("*.md"):
        if other.stem == slug_actual:
            continue
        if f"[[{slug_actual}]]" in other.read_text():
            incoming.append(other.stem)

    lines = [f"Graph for: {slug_actual}\n"]
    lines.append(f"Outgoing links ({len(outgoing)}):")
    for s in outgoing:
        # Get TL;DR from index
        idx = {k: v for k, v in _load_index()}
        tldr = idx.get(s, "")
        lines.append(f"  → [[{s}]]{f' — {tldr}' if tldr else ''}")

    lines.append(f"\nIncoming links ({len(incoming)}):")
    for s in incoming:
        idx = {k: v for k, v in _load_index()}
        tldr = idx.get(s, "")
        lines.append(f"  ← [[{s}]]{f' — {tldr}' if tldr else ''}")

    if not outgoing and not incoming:
        lines.append("  (no connections yet)")

    return {"formatted": "\n".join(lines)}


_OPERATIONS = {
    "search": _op_search,
    "read_note": _op_read_note,
    "list_papers": _op_list_papers,
    "graph": _op_graph,
}

OBSIDIAN_KB_TOOL_SPEC = {
    "name": "obsidian_kb",
    "description": (
        "Search and read your personal Obsidian research vault (compiled paper notes with "
        "structured summaries, backlinks, and metadata).\n\n"
        "This is YOUR curated knowledge base — papers you've already read and summarized. "
        "Check here FIRST before searching external sources.\n\n"
        "Note: vault notes contain structured summaries, not full paper text. "
        "If you need exact paper content (proofs, tables, algorithms), use "
        "hf_papers(operation='read_paper', arxiv_id=...) to read the arxiv source.\n\n"
        "Operations:\n"
        "- search: Find papers in vault by keyword (searches TL;DRs and slugs)\n"
        "- read_note: Read a compiled paper note (summary, contributions, method, results, links)\n"
        "- list_papers: List all indexed papers with TL;DRs\n"
        "- graph: Get a paper's outgoing and incoming backlinks (connections to related work)"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": list(_OPERATIONS.keys()),
                "description": "Operation to execute.",
            },
            "query": {
                "type": "string",
                "description": "Search query. Required for: search.",
            },
            "slug": {
                "type": "string",
                "description": (
                    "Paper note slug (filename without .md, e.g. "
                    "'hao-2024-training-large-language-models-to-reason-in-a-continuous-latent-space'). "
                    "Required for: read_note, graph. Partial matches accepted."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Max results (default 10, max 50).",
            },
        },
        "required": ["operation"],
    },
}


async def obsidian_kb_handler(arguments: dict[str, Any]) -> tuple[str, bool]:
    operation = arguments.get("operation")
    if not operation:
        return "'operation' parameter is required.", False

    handler = _OPERATIONS.get(operation)
    if not handler:
        valid = ", ".join(_OPERATIONS.keys())
        return f"Unknown operation: '{operation}'. Valid: {valid}", False

    limit = min(arguments.get("limit", 10), 50)

    try:
        result = await handler(arguments, limit)
        return result["formatted"], not result.get("isError", False)
    except Exception as e:
        return f"Error in {operation}: {e}", False
