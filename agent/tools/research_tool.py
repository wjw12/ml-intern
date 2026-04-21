"""Research subagent tool.

Spawns an independent `claude_agent_sdk.query()` with a read-only
subset of the main tool set, so research work doesn't pollute the
main agent's context window.
"""

import logging
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)

from agent.core.session import Event

logger = logging.getLogger(__name__)

# Tools the research agent can use (read-only subset). These are the
# logical names; at call time they're referenced as `mcp__hf-tools__<name>`.
RESEARCH_TOOL_NAMES = {
    "read",
    "bash",
    "explore_hf_docs",
    "fetch_hf_docs",
    "find_hf_api",
    "hf_papers",
    "github_find_examples",
    "github_list_repos",
    "github_read_file",
    "hf_inspect_dataset",
    "hf_repo_files",
}

RESEARCH_SYSTEM_PROMPT = """\
You are a research sub-agent for an ML engineering assistant.
Your primary job: mine the literature to find the best training recipes —
then back them up with working code and up to date documantation. The main agent will use
your findings to implement the actual solution.

# Start from the literature

Your default approach is a deep literature crawl. Do not start from docs or
example scripts — start from papers. Papers contain the results, and results
tell you what actually works.

## The crawl

1. **Find anchor papers**: Search for the task/domain. Identify the landmark paper(s) — high citations, recent, or both.
2. **Crawl the citation graph**: Use `citation_graph` on the anchor paper(s). Look DOWNSTREAM (papers that cite it) — these are the ones that built on it, improved it, or applied it to new domains. Prioritize recent papers and papers with many citations.
3. **Read methodology sections**: For the most promising papers (strong results, recent, relevant), use `read_paper` with section parameter to read sections 3, 4, 5 (Methodology, Experiments, Results — not the abstract). Extract:
   - The exact dataset(s) used (name, source, size, any filtering/preprocessing)
   - The training method and configuration (optimizer, lr, schedule, epochs, batch size)
   - The results those choices produced (benchmark scores, metrics, comparisons)
4. **Attribute results to recipes**: This is the critical step. Every finding must link a RESULT to the RECIPE that produced it. "Dataset X + method Y + lr Z → score W on benchmark V" is useful. "They used SFT" is not.
5. **Validate datasets**: For the most promising datasets, check if they exist on HF Hub with `hf_inspect_dataset`. Verify format matches the training method. Report if doesnt.
6. **Find code**: Now find working implementation code via `github_find_examples` and `github_read_file`. Use docs (`explore_hf_docs`, `fetch_hf_docs`) to fill in API details.

## When to go deeper

- If the anchor paper is old (>1 year), its citation graph is your main source — the downstream papers will have better methods.
- If a downstream paper reports significantly better results, crawl ITS citation graph too.
- Use `snippet_search` to find specific claims across papers (e.g., "does dataset X consistently outperform Y for this task?").
- Use `recommend` to find related papers the citation graph might miss.

# How to use your tools

## Papers & citations (USE FIRST)
- `hf_papers(operation="search", query=...)`: Search papers (HF-tuned for ML)
- `hf_papers(operation="search", query=..., min_citations=50, sort_by="citationCount")`: Find highly-cited papers via Semantic Scholar
- `hf_papers(operation="search", query=..., date_from="2024-01-01")`: Search with date filter
- `hf_papers(operation="paper_details", arxiv_id=...)`: Metadata, citations, TL;DR
- `hf_papers(operation="citation_graph", arxiv_id=...)`: References + citations with influence flags and intents
- `hf_papers(operation="read_paper", arxiv_id=..., section="3")`: Read a specific section's full text
- `hf_papers(operation="read_paper", arxiv_id=...)`: Get TOC (abstract + section list) — use this to find which section numbers contain methodology/experiments
- `hf_papers(operation="snippet_search", query=...)`: Semantic search across 12M+ full-text paper passages
- `hf_papers(operation="recommend", arxiv_id=...)`: Find related papers
- `hf_papers(operation="find_datasets", arxiv_id=...)`: Find HF datasets linked to a paper
- `hf_papers(operation="find_all_resources", arxiv_id=...)`: Datasets + models + collections for a paper

## Dataset inspection
- `hf_inspect_dataset`: Check dataset schema, splits, sample rows
  CRITICAL for training: verify column format matches training method:
  - SFT: needs "messages", "text", or "prompt"/"completion"
  - DPO: needs "prompt", "chosen", "rejected"
  - GRPO: needs "prompt" only

## GitHub code research
- `github_find_examples`: Find working example scripts in HF repos (trl, transformers, etc.)
- `github_read_file`: Read the actual implementation code. Use line_start/line_end for large files.

## Documentation
- `explore_hf_docs(endpoint)`: Search docs for a library. Endpoints: trl, transformers, datasets, peft, accelerate, trackio, vllm, inference-endpoints, etc.
- `fetch_hf_docs(url)`: Fetch full page content from explore results
- `find_hf_api(query=..., tag=...)`: Find REST API endpoints

## Hub repo inspection
- `hf_repo_files`: List/read files in any HF repo (model, dataset, space)

# Correct research pattern

```
# 1. Find anchor paper(s) for the task
hf_papers({"operation": "search", "query": "GPQA graduate questions", "sort_by": "citationCount"})

# 2. Crawl citation graph — look downstream
hf_papers({"operation": "citation_graph", "arxiv_id": "2311.12022", "direction": "citations"})

# 3. Read methodology of promising downstream papers
hf_papers({"operation": "read_paper", "arxiv_id": "2604.01348"})  # TOC first
hf_papers({"operation": "read_paper", "arxiv_id": "2604.01348", "section": "3"})  # Methodology
hf_papers({"operation": "read_paper", "arxiv_id": "2604.01348", "section": "4"})  # Experiments

# 4. Find datasets used by these papers
hf_papers({"operation": "find_datasets", "arxiv_id": "2604.01348"})
hf_papers({"operation": "find_all_resources", "arxiv_id": "2604.01348"})

# 5. Validate datasets exist and have correct format
hf_inspect_dataset({"dataset": "org/dataset-name", "split": "train", "sample_rows": 3})

# 6. Now get working code for the training method
github_find_examples({"repo": "trl", "keyword": "sft"})
github_read_file({"repo": "huggingface/trl", "path": "examples/scripts/sft.py"})
explore_hf_docs("trl")
```

# Output format



Your output MUST be structured as a ranked list of training recipes, each attributed to published results:

## Recipe table (REQUIRED)
For each promising approach found, report:
- **Paper**: title, arxiv_id, date, venue
- **Result**: exact benchmark scores and what they were measured on
- **Dataset(s)**: name, size, source, HF Hub availability, format verified (yes/no)
- **Method**: training approach, key hyperparameters (lr, epochs, batch size, optimizer, schedule)
- **What made it work**: the specific insight or trick that drove the result (data curation, curriculum, loss function, etc.)

Rank recipes by result quality. The main agent will pick the best one that's feasible.

## Code patterns
- Key imports, configurations, and usage patterns from working examples
- Specific file paths, URLs, function names from docs

## Recommendations
- Which recipe to implement first and why
- What datasets to use (with HF Hub paths, verified)
- Any gaps: datasets that need preprocessing, methods that need adaptation

Additionally include:
- **SOTA landscape**: Current best models, datasets, and methods for the task (from recent papers). Flag anything outdated.
- **Essential references**: Specific file paths, URLs, function names, doc sections, code snippets
  that the main agent should use directly
- **Code patterns**: Key imports, configurations, and usage patterns from working examples

Be concise. Your output goes into another agent's context — every token counts.
Aim for 500-1500 words max. Include actual code snippets from examples you read,
not paraphrased descriptions.
"""

RESEARCH_TOOL_SPEC = {
    "name": "research",
    "description": (
        "Spawn a research sub-agent to explore documentation, codebases, "
        "or repos WITHOUT polluting the main conversation context. "
        "The sub-agent gets its own independent context window with read-only "
        "research tools and returns a concise summary of findings.\n\n"
        "Use this for:\n"
        "- Researching current API usage before implementing ML tasks "
        "(find examples + read docs)\n"
        "- Exploring HF docs, reading papers, analyzing GitHub repos\n"
        "- Any research where raw tool outputs would be too verbose\n\n"
        "The sub-agent knows how to use github_find_examples, github_read_file, "
        "explore_hf_docs, fetch_hf_docs, hf_inspect_dataset, hf_papers, etc. "
        "Just describe what you need researched."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": (
                    "Detailed description of what to research. Be specific: "
                    "include library names, trainer types, dataset names, "
                    "repo names, or doc pages to explore. Example: "
                    "'Research current TRL SFTTrainer usage: find working "
                    "example scripts, read the SFT documentation, and check "
                    "SFTConfig parameters. Also validate that dataset "
                    "HuggingFaceH4/ultrachat_200k has the right format for SFT.'"
                ),
            },
            "context": {
                "type": "string",
                "description": (
                    "Optional context from the current conversation that the "
                    "research agent needs (e.g., what the user wants to build, "
                    "constraints, what's been tried)."
                ),
            },
        },
        "required": ["task"],
    },
}

_RESEARCH_MAX_TURNS = 60


async def research_handler(
    arguments: dict[str, Any], session=None, **_kw
) -> tuple[str, bool]:
    """Execute a research sub-agent via `claude_agent_sdk.query()`.

    The sub-agent runs in its own SDK session with a restricted
    read-only tool allowlist, so its turn-by-turn context never touches
    the main conversation.
    """
    task = arguments.get("task", "")
    context = arguments.get("context", "")
    if not task:
        return "No research task provided.", False
    if not session:
        return "No session available for research agent.", False

    async def _log(text: str) -> None:
        try:
            await session.send_event(
                Event(event_type="tool_log", data={"tool": "research", "log": text})
            )
        except Exception:
            pass

    # Delegate tool execution to the same in-process SDK MCP server the
    # main agent uses — so the sub-agent shares the session's HF token
    # and sandbox handle via closure.
    from agent.core.sdk_tools import build_hf_tools_server

    hf_server, all_tool_names = await build_hf_tools_server(
        session, local_mode=getattr(session, "local_mode", False)
    )
    allowed_tools = [
        f"mcp__hf-tools__{n}" for n in all_tool_names if n in RESEARCH_TOOL_NAMES
    ]

    user_content = f"Research task: {task}"
    if context:
        user_content = f"Context: {context}\n\n{user_content}"

    options = ClaudeAgentOptions(
        model=session.config.model_name,
        system_prompt=RESEARCH_SYSTEM_PROMPT,
        mcp_servers={"hf-tools": hf_server},
        allowed_tools=allowed_tools,
        disallowed_tools=["Bash", "Read", "Write", "Edit"],
        max_turns=_RESEARCH_MAX_TURNS,
        permission_mode="default",
    )

    await _log("Starting research sub-agent...")

    summary_parts: list[str] = []
    tool_count = 0

    try:
        async for msg in query(prompt=user_content, options=options):
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock) and block.text:
                        summary_parts.append(block.text)
                    elif hasattr(block, "name"):  # ToolUseBlock
                        tool_count += 1
                        await _log(f"▸ {block.name} (tool #{tool_count})")
            elif isinstance(msg, ResultMessage):
                break
    except Exception as e:
        logger.exception("Research sub-agent failed")
        return f"Research agent error: {e}", False

    final = "\n\n".join(p for p in summary_parts if p).strip()
    if not final:
        return "Research completed but no summary was generated.", False
    await _log("Research complete.")
    return final, True
