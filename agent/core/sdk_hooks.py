"""Hooks wired into the Claude Agent SDK:

* `approval_hook` — replaces the old `_needs_approval` gate. For tools that
  should require user confirmation, it returns `permissionDecision: "ask"`,
  which routes to `can_use_tool` where we suspend until the user responds
  via the existing EXEC_APPROVAL op.

* `doom_loop_hook` — replaces the standalone doom-loop detector. Tracks
  recent tool invocations per session and injects a corrective additional
  context when the agent loops.

All hooks are closures over the live `Session` so they can queue events
and resolve pending approval futures.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from agent.config import Config
from agent.core.session import Event, Session
from agent.tools.jobs_tool import CPU_FLAVORS

logger = logging.getLogger(__name__)


# ── Approval gating (mirrors the old _needs_approval) ──────────────────


def _needs_approval(tool_name: str, tool_input: dict, config: Config) -> bool:
    """Return True if a tool invocation should require user confirmation."""
    if config.yolo_mode:
        return False

    if tool_name == "sandbox_create":
        return True

    if tool_name == "hf_jobs":
        operation = tool_input.get("operation", "")
        if operation not in ["run", "uv", "scheduled run", "scheduled uv"]:
            return False
        hardware = (
            tool_input.get("hardware_flavor")
            or tool_input.get("flavor")
            or tool_input.get("hardware")
            or "cpu-basic"
        )
        if hardware in CPU_FLAVORS:
            return config.confirm_cpu_jobs
        return True

    if tool_name == "hf_repo_files":
        if tool_input.get("operation") in ["upload", "delete"]:
            return True

    if tool_name == "hf_repo_git":
        if tool_input.get("operation") in [
            "delete_branch",
            "delete_tag",
            "merge_pr",
            "create_repo",
            "update_repo",
        ]:
            return True

    if tool_name == "hf_private_repos":
        op = tool_input.get("operation", "")
        if op == "upload_file":
            return not config.auto_file_upload
        if op == "create_repo":
            return True

    return False


# ── Doom-loop detection (pure helpers, kept from the old module) ───────


@dataclass(frozen=True)
class _Sig:
    name: str
    args_hash: str


def _hash_args(tool_input: dict) -> str:
    blob = json.dumps(tool_input, sort_keys=True, default=str)
    return hashlib.md5(blob.encode()).hexdigest()[:12]


def _identical_consecutive(sigs: list[_Sig], threshold: int = 3) -> str | None:
    if len(sigs) < threshold:
        return None
    count = 1
    for i in range(1, len(sigs)):
        if sigs[i] == sigs[i - 1]:
            count += 1
            if count >= threshold:
                return sigs[i].name
        else:
            count = 1
    return None


def _repeating_sequence(sigs: list[_Sig]) -> list[_Sig] | None:
    n = len(sigs)
    for seq_len in range(2, 6):
        min_required = seq_len * 2
        if n < min_required:
            continue
        tail = sigs[-min_required:]
        pattern = tail[:seq_len]
        reps = 0
        for start in range(n - seq_len, -1, -seq_len):
            chunk = sigs[start : start + seq_len]
            if chunk == pattern:
                reps += 1
            else:
                break
        if reps >= 2:
            return pattern
    return None


@dataclass
class _LoopState:
    history: deque = field(default_factory=lambda: deque(maxlen=30))


def build_hooks(session: Session, approval_manager: "ApprovalManager") -> dict:
    """Build the hook map to pass to ClaudeAgentOptions(hooks=...)."""

    loop_state = _LoopState()

    async def pre_tool_use(input_data: dict, tool_use_id: str | None, context: Any):
        tool_name = input_data.get("tool_name", "") or ""
        tool_input = input_data.get("tool_input", {}) or {}

        # --- Doom-loop detection
        loop_state.history.append(_Sig(name=tool_name, args_hash=_hash_args(tool_input)))
        sigs = list(loop_state.history)
        looped = _identical_consecutive(sigs) or (
            _repeating_sequence(sigs) and "pattern"
        )
        if looped:
            await session.send_event(
                Event(
                    event_type="tool_log",
                    data={
                        "tool": "system",
                        "log": "Doom loop detected — injecting corrective prompt",
                    },
                )
            )
            loop_state.history.clear()
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "additionalContext": (
                        "[SYSTEM: DOOM LOOP DETECTED] You've been repeating the same "
                        "tool calls without progress. Stop, step back, and try a "
                        "fundamentally different strategy — different tool, different "
                        "arguments, or ask the user for guidance."
                    ),
                }
            }

        # --- Approval gating
        if not _needs_approval(tool_name, tool_input, session.config):
            return {}

        # Ask the user via the existing approval_required → EXEC_APPROVAL flow.
        decision = await approval_manager.request(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_use_id=tool_use_id or "",
        )
        if decision.approved:
            # If the user edited an inline script, mutate tool_input in place —
            # the SDK forwards our dict back to the tool invocation.
            if decision.edited_script and "script" in tool_input:
                tool_input["script"] = decision.edited_script
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "allow",
                    "permissionDecisionReason": "approved by user",
                }
            }
        reason = "cancelled by user"
        if decision.feedback:
            reason = f"cancelled by user: {decision.feedback}"
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }

    return {"PreToolUse": [{"matcher": "*", "hooks": [pre_tool_use]}]}


# ── Approval manager: bridges SDK hook to the existing approval op ─────


@dataclass
class ApprovalDecision:
    approved: bool
    feedback: str | None = None
    edited_script: str | None = None


class ApprovalManager:
    """Mediates between the SDK's PreToolUse hook and the UI's EXEC_APPROVAL op.

    * The hook calls `request(...)`, which emits an `approval_required` event
      and suspends on a Future keyed by `tool_use_id`.
    * When the UI submits an EXEC_APPROVAL op, `resolve(...)` wakes the hook.
    """

    def __init__(self, session: Session) -> None:
        self._session = session
        self._pending: dict[str, asyncio.Future[ApprovalDecision]] = {}
        # Also kept on the session so abandonment logic can inspect it.
        session.pending_approval = {"tool_calls": []}

    async def request(
        self, tool_name: str, tool_input: dict, tool_use_id: str
    ) -> ApprovalDecision:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[ApprovalDecision] = loop.create_future()
        self._pending[tool_use_id] = fut

        entry = {
            "tool": tool_name,
            "tool_call_id": tool_use_id,
            "arguments": tool_input,
        }
        self._session.pending_approval["tool_calls"].append(entry)

        await self._session.send_event(
            Event(
                event_type="approval_required",
                data={
                    "count": 1,
                    "tools": [entry],
                },
            )
        )

        try:
            return await fut
        finally:
            self._pending.pop(tool_use_id, None)
            self._session.pending_approval["tool_calls"] = [
                t
                for t in self._session.pending_approval["tool_calls"]
                if t["tool_call_id"] != tool_use_id
            ]

    def resolve_all(self, approvals: list[dict]) -> None:
        """Resolve pending approvals from an EXEC_APPROVAL operation."""
        for a in approvals:
            tid = a.get("tool_call_id", "")
            fut = self._pending.get(tid)
            if fut is None or fut.done():
                continue
            fut.set_result(
                ApprovalDecision(
                    approved=bool(a.get("approved", False)),
                    feedback=a.get("feedback"),
                    edited_script=a.get("edited_script"),
                )
            )

    def abandon_all(self) -> None:
        """Reject any outstanding approvals (e.g. user sent a new message)."""
        for tid, fut in list(self._pending.items()):
            if not fut.done():
                fut.set_result(
                    ApprovalDecision(
                        approved=False,
                        feedback="Task abandoned — user continued the conversation.",
                    )
                )
        self._pending.clear()
