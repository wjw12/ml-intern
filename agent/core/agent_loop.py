"""Agent loop — now a thin dispatcher over `SdkRunner`.

The Claude Agent SDK owns the per-turn LLM interaction (streaming,
tool execution, retries, compaction). This module's job is to:

* Initialise the runner on startup.
* Translate `Operation` messages from the submission queue into runner
  calls (`USER_INPUT` → `run_turn`, `EXEC_APPROVAL` → resolve pending
  approvals, `INTERRUPT` → cancel, `SHUTDOWN` → exit).
* Emit lifecycle events (`ready`, `shutdown`, `error`) the UI expects.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from agent.config import Config
from agent.core.sdk_runner import SdkRunner
from agent.core.session import Event, OpType, Session

logger = logging.getLogger(__name__)


async def _resolve_hf_username(hf_token: str | None) -> str:
    """Best-effort HF username lookup for the system prompt header.

    Uses `curl -4` to dodge the IPv6 Happy-Eyeballs hang we've seen in
    Python HTTP clients against huggingface.co.
    """
    if not hf_token:
        return "unknown"
    try:
        proc = await asyncio.create_subprocess_exec(
            "curl",
            "-s",
            "-4",
            "-m",
            "5",
            "-H",
            f"Authorization: Bearer {hf_token}",
            "https://huggingface.co/api/whoami-v2",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=7)
        if proc.returncode == 0 and stdout:
            import json

            data = json.loads(stdout)
            return data.get("name") or "unknown"
    except Exception as e:  # noqa: BLE001
        logger.warning("HF whoami failed: %s", e)
    return "unknown"


async def _run_user_input(runner: SdkRunner, session: Session, text: str) -> None:
    try:
        await runner.run_turn(text)
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("run_turn failed")
        await session.send_event(Event(event_type="error", data={"error": str(e)}))


async def process_submission(
    session: Session, runner: SdkRunner, submission: Any
) -> bool:
    """Dispatch one submission. Returns False on shutdown."""
    op = submission.operation
    logger.debug("op=%s", op.op_type.value)

    if op.op_type == OpType.USER_INPUT:
        text = (op.data or {}).get("text", "")
        await _run_user_input(runner, session, text)
        return True

    if op.op_type == OpType.EXEC_APPROVAL:
        approvals = (op.data or {}).get("approvals", [])
        runner.approval_manager.resolve_all(approvals)
        return True

    if op.op_type == OpType.INTERRUPT:
        session.cancel()
        await runner.interrupt()
        return True

    if op.op_type == OpType.COMPACT:
        # Compaction is owned by the Claude Code CLI; nothing to do here.
        # The user sees a `compacted` event when the SDK auto-compacts.
        await session.send_event(
            Event(
                event_type="tool_log",
                data={
                    "tool": "system",
                    "log": "Manual compaction is handled by the Claude Agent SDK; this session auto-compacts as needed.",
                },
            )
        )
        return True

    if op.op_type == OpType.SHUTDOWN:
        if session.config.save_sessions:
            session.save_and_upload_detached(session.config.session_dataset_repo)
        session.is_running = False
        await session.send_event(Event(event_type="shutdown"))
        return False

    logger.warning("Unknown op: %s", op.op_type)
    return True


async def submission_loop(
    submission_queue: asyncio.Queue,
    event_queue: asyncio.Queue,
    config: Config,
    session_holder: list | None = None,
    hf_token: str | None = None,
    local_mode: bool = False,
    stream: bool = True,
) -> None:
    """Main agent loop — owns a Session + SdkRunner and dispatches submissions."""
    session = Session(
        event_queue,
        config=config,
        hf_token=hf_token,
        local_mode=local_mode,
        stream=stream,
    )
    if session_holder is not None:
        session_holder[0] = session

    runner = SdkRunner(session, local_mode=local_mode)

    if config.save_sessions:
        Session.retry_failed_uploads_detached(
            directory="session_logs", repo_id=config.session_dataset_repo
        )

    try:
        hf_username = await _resolve_hf_username(hf_token)
        tool_count = await runner.start(hf_username=hf_username)

        await session.send_event(
            Event(
                event_type="ready",
                data={"message": "Agent initialized", "tool_count": tool_count},
            )
        )

        while session.is_running:
            submission = await submission_queue.get()
            try:
                should_continue = await process_submission(session, runner, submission)
                if not should_continue:
                    break
            except asyncio.CancelledError:
                logger.warning("Agent loop cancelled")
                break
            except Exception as e:  # noqa: BLE001
                logger.exception("Error in agent loop")
                await session.send_event(
                    Event(event_type="error", data={"error": str(e)})
                )
    finally:
        if session.config.save_sessions and session.is_running:
            try:
                session.save_and_upload_detached(session.config.session_dataset_repo)
            except Exception as e:  # noqa: BLE001
                logger.error("Emergency save failed: %s", e)
        await runner.close()
        logger.info("Agent loop exited")
