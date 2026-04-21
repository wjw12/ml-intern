"""Wrap the existing HF tool handlers as a Claude Agent SDK MCP server.

Each session builds its own in-process MCP server so the tool closures
can capture a reference to the live Session (which carries the HF token,
sandbox handle, event queue, and cancellation flag).
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from claude_agent_sdk import create_sdk_mcp_server, tool

from agent.tools.dataset_tools import (
    HF_INSPECT_DATASET_TOOL_SPEC,
    hf_inspect_dataset_handler,
)
from agent.tools.docs_tools import (
    EXPLORE_HF_DOCS_TOOL_SPEC,
    HF_DOCS_FETCH_TOOL_SPEC,
    _get_api_search_tool_spec,
    explore_hf_docs_handler,
    hf_docs_fetch_handler,
    search_openapi_handler,
)
from agent.tools.github_find_examples import (
    GITHUB_FIND_EXAMPLES_TOOL_SPEC,
    github_find_examples_handler,
)
from agent.tools.github_list_repos import (
    GITHUB_LIST_REPOS_TOOL_SPEC,
    github_list_repos_handler,
)
from agent.tools.github_read_file import (
    GITHUB_READ_FILE_TOOL_SPEC,
    github_read_file_handler,
)
from agent.tools.hf_repo_files_tool import HF_REPO_FILES_TOOL_SPEC, hf_repo_files_handler
from agent.tools.hf_repo_git_tool import HF_REPO_GIT_TOOL_SPEC, hf_repo_git_handler
from agent.tools.jobs_tool import HF_JOBS_TOOL_SPEC, hf_jobs_handler
from agent.tools.papers_tool import HF_PAPERS_TOOL_SPEC, hf_papers_handler
from agent.tools.plan_tool import PLAN_TOOL_SPEC, plan_tool_handler
from agent.tools.research_tool import RESEARCH_TOOL_SPEC, research_handler

logger = logging.getLogger(__name__)

# Shape of a legacy handler: async (args, session=?) -> (text, success)
Handler = Callable[..., Awaitable[tuple[str, bool]]]


def _wrap_handler(handler: Handler, session: Any) -> Callable[[dict], Awaitable[dict]]:
    """Adapt a legacy `(args, session=?) -> (str, bool)` handler to the SDK's
    `(args) -> {"content": [...]}` signature, injecting the session via closure.

    We pass `session` by keyword only when the handler accepts it, to keep
    the wrapper compatible with handlers that don't.
    """
    import inspect

    sig = inspect.signature(handler)
    accepts_session = "session" in sig.parameters

    async def wrapped(args: dict[str, Any]) -> dict[str, Any]:
        try:
            if accepts_session:
                text, ok = await handler(args, session=session)
            else:
                text, ok = await handler(args)
        except Exception as e:  # noqa: BLE001
            logger.exception("Tool handler raised")
            return {
                "content": [{"type": "text", "text": f"Tool error: {e}"}],
                "isError": True,
            }
        return {
            "content": [{"type": "text", "text": text or ""}],
            "isError": not ok,
        }

    return wrapped


def _register(name: str, description: str, schema: dict, handler: Callable):
    """Apply the @tool decorator with the tool's declared JSON schema."""
    return tool(name, description, schema)(handler)


# Built-in tools that ship with every session.
# (Paper tools, docs, github, dataset inspection, planning, jobs, repo mgmt.)
_BASE_TOOL_DEFS: list[tuple[str, dict]] = [
    (RESEARCH_TOOL_SPEC["name"], RESEARCH_TOOL_SPEC),
    (EXPLORE_HF_DOCS_TOOL_SPEC["name"], EXPLORE_HF_DOCS_TOOL_SPEC),
    (HF_DOCS_FETCH_TOOL_SPEC["name"], HF_DOCS_FETCH_TOOL_SPEC),
    (HF_PAPERS_TOOL_SPEC["name"], HF_PAPERS_TOOL_SPEC),
    (HF_INSPECT_DATASET_TOOL_SPEC["name"], HF_INSPECT_DATASET_TOOL_SPEC),
    (PLAN_TOOL_SPEC["name"], PLAN_TOOL_SPEC),
    (HF_JOBS_TOOL_SPEC["name"], HF_JOBS_TOOL_SPEC),
    (HF_REPO_FILES_TOOL_SPEC["name"], HF_REPO_FILES_TOOL_SPEC),
    (HF_REPO_GIT_TOOL_SPEC["name"], HF_REPO_GIT_TOOL_SPEC),
    (GITHUB_FIND_EXAMPLES_TOOL_SPEC["name"], GITHUB_FIND_EXAMPLES_TOOL_SPEC),
    (GITHUB_LIST_REPOS_TOOL_SPEC["name"], GITHUB_LIST_REPOS_TOOL_SPEC),
    (GITHUB_READ_FILE_TOOL_SPEC["name"], GITHUB_READ_FILE_TOOL_SPEC),
]

_BASE_HANDLERS: dict[str, Handler] = {
    RESEARCH_TOOL_SPEC["name"]: research_handler,
    EXPLORE_HF_DOCS_TOOL_SPEC["name"]: explore_hf_docs_handler,
    HF_DOCS_FETCH_TOOL_SPEC["name"]: hf_docs_fetch_handler,
    HF_PAPERS_TOOL_SPEC["name"]: hf_papers_handler,
    HF_INSPECT_DATASET_TOOL_SPEC["name"]: hf_inspect_dataset_handler,
    PLAN_TOOL_SPEC["name"]: plan_tool_handler,
    HF_JOBS_TOOL_SPEC["name"]: hf_jobs_handler,
    HF_REPO_FILES_TOOL_SPEC["name"]: hf_repo_files_handler,
    HF_REPO_GIT_TOOL_SPEC["name"]: hf_repo_git_handler,
    GITHUB_FIND_EXAMPLES_TOOL_SPEC["name"]: github_find_examples_handler,
    GITHUB_LIST_REPOS_TOOL_SPEC["name"]: github_list_repos_handler,
    GITHUB_READ_FILE_TOOL_SPEC["name"]: github_read_file_handler,
}


async def build_hf_tools_server(session: Any, local_mode: bool = False):
    """Build a per-session SDK MCP server wrapping every HF tool.

    Returns the MCP server object and the list of tool names it exposes
    (callers use the latter to set `allowed_tools`).
    """
    tool_fns = []
    tool_names: list[str] = []

    # Base tools
    for name, spec in _BASE_TOOL_DEFS:
        handler = _BASE_HANDLERS[name]
        wrapped = _wrap_handler(handler, session)
        decorated = _register(name, spec["description"], spec["parameters"], wrapped)
        tool_fns.append(decorated)
        tool_names.append(name)

    # OpenAPI search tool (async-initialized)
    try:
        openapi_spec = await _get_api_search_tool_spec()
        wrapped = _wrap_handler(search_openapi_handler, session)
        decorated = _register(
            openapi_spec["name"],
            openapi_spec["description"],
            openapi_spec["parameters"],
            wrapped,
        )
        tool_fns.append(decorated)
        tool_names.append(openapi_spec["name"])
    except Exception as e:  # noqa: BLE001
        logger.warning("Skipping OpenAPI search tool: %s", e)

    # Sandbox vs local mode — filesystem + bash tools
    if local_mode:
        from agent.tools.local_tools import _HANDLERS, _LOCAL_TOOL_SPECS  # type: ignore

        for name, spec in _LOCAL_TOOL_SPECS.items():
            handler = _HANDLERS[name]
            wrapped = _wrap_handler(handler, session)
            decorated = _register(name, spec["description"], spec["parameters"], wrapped)
            tool_fns.append(decorated)
            tool_names.append(name)
    else:
        from agent.tools.sandbox_client import Sandbox
        from agent.tools.sandbox_tool import (
            SANDBOX_CREATE_TOOL_SPEC,
            _make_tool_handler,
            sandbox_create_handler,
        )

        wrapped = _wrap_handler(sandbox_create_handler, session)
        decorated = _register(
            SANDBOX_CREATE_TOOL_SPEC["name"],
            SANDBOX_CREATE_TOOL_SPEC["description"],
            SANDBOX_CREATE_TOOL_SPEC["parameters"],
            wrapped,
        )
        tool_fns.append(decorated)
        tool_names.append(SANDBOX_CREATE_TOOL_SPEC["name"])

        for op_name in Sandbox.TOOLS:
            op_spec = Sandbox.TOOLS[op_name]
            wrapped = _wrap_handler(_make_tool_handler(op_name), session)
            decorated = _register(
                op_name, op_spec["description"], op_spec["parameters"], wrapped
            )
            tool_fns.append(decorated)
            tool_names.append(op_name)

    server = create_sdk_mcp_server(
        name="hf-tools", version="1.0.0", tools=tool_fns
    )
    logger.info("Built hf-tools MCP server with %d tools", len(tool_names))
    return server, tool_names


# Names of tools that are read-only and safe for the research sub-agent.
RESEARCH_TOOL_NAMES: set[str] = {
    "read",
    "bash",
    EXPLORE_HF_DOCS_TOOL_SPEC["name"],
    HF_DOCS_FETCH_TOOL_SPEC["name"],
    "find_hf_api",
    HF_PAPERS_TOOL_SPEC["name"],
    GITHUB_FIND_EXAMPLES_TOOL_SPEC["name"],
    GITHUB_LIST_REPOS_TOOL_SPEC["name"],
    GITHUB_READ_FILE_TOOL_SPEC["name"],
    HF_INSPECT_DATASET_TOOL_SPEC["name"],
    HF_REPO_FILES_TOOL_SPEC["name"],
}
