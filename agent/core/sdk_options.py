"""Assemble a `ClaudeAgentOptions` for a single ML Intern session.

Responsibilities:
* Render the system prompt (YAML + Jinja) with session context.
* Register the per-session SDK MCP server (`hf-tools`) plus any configured
  remote MCP servers (e.g. the HF MCP endpoint).
* Wire the PreToolUse hook (approval + doom-loop detection).
* Turn the `Config` into SDK-level knobs (permission mode, max_turns,
  allowed/disallowed tools).
"""

from __future__ import annotations

import logging
import os
import zoneinfo
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from claude_agent_sdk import ClaudeAgentOptions
from jinja2 import Template

from agent.config import Config, RemoteMCPServer, StdioMCPServer
from agent.core.sdk_hooks import ApprovalManager, build_hooks
from agent.core.sdk_tools import build_hf_tools_server
from agent.core.session import Session

logger = logging.getLogger(__name__)

_PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "system_prompt_v3.yaml"

# Tool names from the Claude Code CLI's built-in toolset that we replace
# with our own sandbox/local equivalents. Disallow them so Claude uses ours.
_CC_BUILTIN_DUPLICATES = ["Bash", "Read", "Write", "Edit"]


def _render_system_prompt(
    tool_names: list[str], hf_username: str, local_mode: bool, cwd: str
) -> str:
    prompt_data = yaml.safe_load(_PROMPT_FILE.read_text())
    template = Template(prompt_data.get("system_prompt", ""))
    body = template.render(tools=tool_names, num_tools=len(tool_names))

    tz = zoneinfo.ZoneInfo(os.environ.get("ML_INTERN_TZ", "UTC"))
    now = datetime.now(tz)
    context = (
        f"\n\n[Session context: Date={now.strftime('%Y-%m-%d')}, "
        f"Time={now.strftime('%H:%M:%S')}, Timezone={now.strftime('%Z')}, "
        f"User={hf_username}, Tools={len(tool_names)}]"
    )

    if local_mode:
        context = (
            f"\n\n# CLI / Local mode\n\n"
            f"You are running as a local CLI tool on the user's machine. "
            f"There is NO sandbox — bash, read, write, and edit operate directly "
            f"on the local filesystem.\n\n"
            f"Working directory: {cwd}\n"
            f"Use absolute paths or paths relative to the working directory. "
            f"The sandbox_create tool is NOT available. Run code directly with bash."
        ) + context

    return body + context


def _mcp_servers_from_config(
    config: Config, hf_token: str | None = None
) -> dict[str, Any]:
    """Translate our Config.mcpServers into the dict shape the SDK expects."""
    out: dict[str, Any] = {}
    for name, server in config.mcpServers.items():
        if isinstance(server, RemoteMCPServer):
            spec: dict[str, Any] = {"type": server.transport, "url": server.url}
            headers = dict(server.headers) if server.headers else {}
            if hf_token:
                headers.setdefault("Authorization", f"Bearer {hf_token}")
            if headers:
                spec["headers"] = headers
            out[name] = spec
        elif isinstance(server, StdioMCPServer):
            out[name] = {
                "type": "stdio",
                "command": server.command,
                "args": list(server.args),
                "env": dict(server.env),
            }
    return out


async def build_options(
    session: Session,
    approval_manager: ApprovalManager,
    local_mode: bool = False,
    hf_username: str = "unknown",
) -> tuple[ClaudeAgentOptions, list[str]]:
    """Build ClaudeAgentOptions + return the list of HF tool names.

    The caller uses the tool-name list to decide what goes into
    `allowed_tools` — anything on it is a local SDK MCP tool and thus
    prefixed with `mcp__hf-tools__...` in the SDK's allowlist syntax.
    """
    cwd = os.getcwd()

    hf_server, hf_tool_names = await build_hf_tools_server(session, local_mode=local_mode)

    mcp_servers: dict[str, Any] = {"hf-tools": hf_server}
    mcp_servers.update(_mcp_servers_from_config(session.config, hf_token=session.hf_token))

    # Pre-approve our HF tools (we gate the sensitive ones via the hook instead).
    allowed_tools = [f"mcp__hf-tools__{n}" for n in hf_tool_names]

    system_prompt = _render_system_prompt(hf_tool_names, hf_username, local_mode, cwd)

    permission_mode = "acceptEdits" if session.config.yolo_mode else "default"
    max_turns = (
        None if session.config.max_iterations == -1 else session.config.max_iterations
    )

    hooks = build_hooks(session, approval_manager)

    options = ClaudeAgentOptions(
        model=session.config.model_name,
        system_prompt=system_prompt,
        cwd=cwd,
        mcp_servers=mcp_servers,
        allowed_tools=allowed_tools,
        disallowed_tools=_CC_BUILTIN_DUPLICATES,
        hooks=hooks,
        permission_mode=permission_mode,
        max_turns=max_turns,
        resume=session.config.resume_session_id,
    )
    return options, hf_tool_names
