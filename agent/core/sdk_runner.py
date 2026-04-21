"""Thin adapter that owns the `ClaudeSDKClient` lifecycle for a session
and translates SDK message objects into the existing `Event` stream the
UI already understands.

Mapping:
  AssistantMessage + TextBlock     → assistant_message
  AssistantMessage + ToolUseBlock  → tool_call
  UserMessage + ToolResultBlock    → tool_output
  ResultMessage                    → turn_complete
  SystemMessage (compacted)        → compacted

Cancellation is cooperative: when `session._cancelled` is set, we call
`client.interrupt()` and break out of the iteration.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeSDKClient,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from agent.core.sdk_hooks import ApprovalManager
from agent.core.sdk_options import build_options
from agent.core.session import Event, Session

logger = logging.getLogger(__name__)


class SdkRunner:
    """Drives a `ClaudeSDKClient` on behalf of a `Session`.

    The runner is constructed eagerly but `start()` must be awaited before
    `run_turn()` will work — it builds the options (which needs async) and
    enters the client's async context manager.
    """

    def __init__(self, session: Session, local_mode: bool = False) -> None:
        self._session = session
        self._local_mode = local_mode
        self._client: ClaudeSDKClient | None = None
        self._approval_manager = ApprovalManager(session)
        self._entered = False

    @property
    def approval_manager(self) -> ApprovalManager:
        return self._approval_manager

    async def start(self, hf_username: str = "unknown") -> int:
        """Initialise the client; returns the count of registered HF tools."""
        options, tool_names = await build_options(
            self._session,
            self._approval_manager,
            local_mode=self._local_mode,
            hf_username=hf_username,
        )
        self._client = ClaudeSDKClient(options=options)
        await self._client.__aenter__()
        self._entered = True
        return len(tool_names)

    async def close(self) -> None:
        if self._client is None or not self._entered:
            return
        try:
            await self._client.__aexit__(None, None, None)
        except Exception as e:  # noqa: BLE001
            logger.warning("Error closing Claude SDK client: %s", e)
        self._client = None
        self._entered = False

    async def interrupt(self) -> None:
        if self._client is None:
            return
        try:
            await self._client.interrupt()
        except Exception as e:  # noqa: BLE001
            logger.warning("interrupt() failed: %s", e)

    async def run_turn(self, text: str) -> None:
        """Send user text, pump messages out as events, return on turn end."""
        if self._client is None:
            raise RuntimeError("SdkRunner not started")

        self._session.reset_cancel()
        self._approval_manager.abandon_all()

        await self._session.send_event(
            Event(event_type="processing", data={"message": "Processing user input"})
        )

        await self._client.query(text)

        try:
            async for msg in self._client.receive_response():
                if self._session.is_cancelled:
                    await self.interrupt()
                    await self._session.send_event(Event(event_type="interrupted"))
                    return
                await self._dispatch(msg)
        except asyncio.CancelledError:
            await self.interrupt()
            raise

        self._session.increment_turn()
        await self._session.auto_save_if_needed()

    # ── Message → Event translation ───────────────────────────────────

    async def _dispatch(self, msg: Any) -> None:
        if isinstance(msg, AssistantMessage):
            await self._handle_assistant(msg)
        elif isinstance(msg, UserMessage):
            await self._handle_user(msg)
        elif isinstance(msg, ResultMessage):
            await self._handle_result(msg)
        elif isinstance(msg, SystemMessage):
            await self._handle_system(msg)
        else:
            logger.debug("Unhandled SDK message type: %s", type(msg).__name__)

    async def _handle_assistant(self, msg: AssistantMessage) -> None:
        for block in msg.content:
            if isinstance(block, TextBlock):
                text = block.text or ""
                if text:
                    await self._session.send_event(
                        Event(
                            event_type="assistant_message",
                            data={"content": text},
                        )
                    )
            elif isinstance(block, ToolUseBlock):
                await self._session.send_event(
                    Event(
                        event_type="tool_call",
                        data={
                            "tool": _strip_mcp_prefix(block.name),
                            "arguments": block.input or {},
                            "tool_call_id": block.id,
                        },
                    )
                )

    async def _handle_user(self, msg: UserMessage) -> None:
        # The SDK surfaces tool results as UserMessage blocks.
        content = msg.content
        if not isinstance(content, list):
            return
        for block in content:
            if isinstance(block, ToolResultBlock):
                output = _stringify_tool_result(block.content)
                success = not getattr(block, "is_error", False)
                await self._session.send_event(
                    Event(
                        event_type="tool_output",
                        data={
                            "tool_call_id": block.tool_use_id,
                            "tool": "",
                            "output": output,
                            "success": success,
                        },
                    )
                )

    async def _handle_result(self, msg: ResultMessage) -> None:
        await self._session.send_event(
            Event(event_type="assistant_stream_end", data={})
        )
        await self._session.send_event(
            Event(
                event_type="turn_complete",
                data={
                    "tokens": getattr(msg, "total_cost_usd", None),
                    "duration_ms": getattr(msg, "duration_ms", None),
                    "num_turns": getattr(msg, "num_turns", None),
                },
            )
        )

    async def _handle_system(self, msg: SystemMessage) -> None:
        # Surface compaction events if the SDK reports them.
        subtype = getattr(msg, "subtype", "") or ""
        if "compact" in subtype.lower():
            await self._session.send_event(
                Event(event_type="compacted", data=dict(getattr(msg, "data", {}) or {}))
            )


def _strip_mcp_prefix(name: str) -> str:
    """Turn `mcp__hf-tools__research` back into `research` for display."""
    if name.startswith("mcp__"):
        parts = name.split("__", 2)
        if len(parts) == 3:
            return parts[2]
    return name


def _stringify_tool_result(content: Any) -> str:
    """Tool results come as a string, a list of content blocks, or None."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                if b.get("type") == "text":
                    parts.append(b.get("text", ""))
                else:
                    parts.append(json.dumps(b))
            else:
                text = getattr(b, "text", None)
                parts.append(text if text is not None else str(b))
        return "\n".join(p for p in parts if p)
    return str(content)
