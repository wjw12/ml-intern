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
    session: Session, runner: SdkRunner, submission: Any,
    run_task: list[asyncio.Task | None] | None = None,
) -> bool:
    """Dispatch one submission. Returns False on shutdown.

    ``run_task`` is a single-element list holding the background task for
    the current ``run_turn`` call (if any).  USER_INPUT spawns a new task
    instead of awaiting directly so the loop stays free to dequeue
    EXEC_APPROVAL and INTERRUPT ops while the turn is in progress.
    """
    if run_task is None:
        run_task = [None]

    op = submission.operation
    logger.debug("op=%s", op.op_type.value)

    if op.op_type == OpType.USER_INPUT:
        # Wait for any prior turn to finish before starting a new one.
        if run_task[0] is not None and not run_task[0].done():
            await run_task[0]
        text = (op.data or {}).get("text", "")
        run_task[0] = asyncio.create_task(_run_user_input(runner, session, text))
        return True

    if op.op_type == OpType.EXEC_APPROVAL:
        approvals = (op.data or {}).get("approvals", [])
        runner.approval_manager.resolve_all(approvals)
        return True

    if op.op_type == OpType.INTERRUPT:
        session.cancel()
        await runner.interrupt()
        if run_task[0] is not None and not run_task[0].done():
            run_task[0].cancel()
            try:
                await run_task[0]
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        return True

    if op.op_type == OpType.COMPACT:
        # Send /compact as a query to the underlying Claude Code CLI.
        if run_task[0] is not None and not run_task[0].done():
            await run_task[0]
        run_task[0] = asyncio.create_task(_run_user_input(runner, session, "/compact"))
        return True

    if op.op_type == OpType.CONTEXT_USAGE:
        usage = await runner.get_context_usage()
        event_data = dict(usage) if usage else {}
        # Pass through source marker if present
        source = (op.data or {}).get("source")
        if source:
            event_data["source"] = source
        await session.send_event(
            Event(event_type="context_usage", data=event_data)
        )
        return True

    if op.op_type == OpType.SHUTDOWN:
        # Wait for any in-flight turn to finish before shutting down.
        if run_task[0] is not None and not run_task[0].done():
            run_task[0].cancel()
            try:
                await run_task[0]
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
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

    # Shared mutable slot so process_submission can track the background task.
    run_task: list[asyncio.Task | None] = [None]

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
                should_continue = await process_submission(
                    session, runner, submission, run_task
                )
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
        # Ensure the background turn task is cleaned up.
        if run_task[0] is not None and not run_task[0].done():
            run_task[0].cancel()
            try:
                await run_task[0]
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        if session.config.save_sessions and session.is_running:
            try:
                session.save_and_upload_detached(session.config.session_dataset_repo)
            except Exception as e:  # noqa: BLE001
                logger.error("Emergency save failed: %s", e)
        await runner.close()
        logger.info("Agent loop exited")
