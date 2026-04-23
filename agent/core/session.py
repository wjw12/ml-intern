"""Session state for a single ML Intern conversation.

The Claude Agent SDK owns the LLM conversation history, so this class no
longer holds a `ContextManager` or tool router. It's now a thin carrier
for the event queue, cancellation flag, trajectory logging, sandbox
handle, and configuration.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from agent.config import Config

logger = logging.getLogger(__name__)


class OpType(Enum):
    USER_INPUT = "user_input"
    EXEC_APPROVAL = "exec_approval"
    INTERRUPT = "interrupt"
    COMPACT = "compact"
    CONTEXT_USAGE = "context_usage"
    SHUTDOWN = "shutdown"
    RESUME = "resume"


@dataclass
class Event:
    event_type: str
    data: Optional[dict[str, Any]] = None


class Session:
    """State container for one user session.

    The agent loop runs Claude via `SdkRunner`; this class only holds
    cross-cutting state that the tools and UI need.
    """

    def __init__(
        self,
        event_queue: asyncio.Queue,
        config: Config,
        hf_token: str | None = None,
        local_mode: bool = False,
        stream: bool = True,
    ) -> None:
        self.event_queue = event_queue
        self.config = config
        self.hf_token = hf_token
        self.local_mode = local_mode
        self.stream = stream

        self.session_id = str(uuid.uuid4())
        self.is_running = True
        self._cancelled = asyncio.Event()

        # Tool-scoped state
        self.sandbox = None
        self._running_job_ids: set[str] = set()

        # Populated by the approval manager when a tool is waiting for approval.
        self.pending_approval: Optional[dict[str, Any]] = None

        # Trajectory logging
        self.logged_events: list[dict] = []
        self.session_start_time = datetime.now().isoformat()
        self.turn_count: int = 0
        self.last_auto_save_turn: int = 0

    # ── Event plumbing ─────────────────────────────────────────────────

    async def send_event(self, event: Event) -> None:
        await self.event_queue.put(event)
        self.logged_events.append(
            {
                "timestamp": datetime.now().isoformat(),
                "event_type": event.event_type,
                "data": event.data,
            }
        )

    # ── Cancellation ───────────────────────────────────────────────────

    def cancel(self) -> None:
        self._cancelled.set()

    def reset_cancel(self) -> None:
        self._cancelled.clear()

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    # ── Config helpers ────────────────────────────────────────────────

    def update_model(self, model_name: str) -> None:
        """Switch the active model. Takes effect on the next `SdkRunner.start()`."""
        self.config.model_name = model_name

    # ── Turn accounting + auto-save ───────────────────────────────────

    def increment_turn(self) -> None:
        self.turn_count += 1

    async def auto_save_if_needed(self) -> None:
        if not self.config.save_sessions:
            return
        interval = self.config.auto_save_interval
        if interval <= 0:
            return
        if self.turn_count - self.last_auto_save_turn >= interval:
            logger.info("Auto-saving session (turn %d)...", self.turn_count)
            self.save_and_upload_detached(self.config.session_dataset_repo)
            self.last_auto_save_turn = self.turn_count

    # ── Trajectory logging ────────────────────────────────────────────

    def get_trajectory(self) -> dict:
        return {
            "session_id": self.session_id,
            "session_start_time": self.session_start_time,
            "session_end_time": datetime.now().isoformat(),
            "model_name": self.config.model_name,
            "events": self.logged_events,
        }

    def save_trajectory_local(
        self,
        directory: str = "session_logs",
        upload_status: str = "pending",
        dataset_url: Optional[str] = None,
    ) -> Optional[str]:
        try:
            log_dir = Path(directory)
            log_dir.mkdir(parents=True, exist_ok=True)
            trajectory = self.get_trajectory()
            trajectory["upload_status"] = upload_status
            trajectory["upload_url"] = dataset_url
            trajectory["last_save_time"] = datetime.now().isoformat()
            filename = (
                f"session_{self.session_id}_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            filepath = log_dir / filename
            with open(filepath, "w") as f:
                json.dump(trajectory, f, indent=2, default=str)
            return str(filepath)
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to save session locally: %s", e)
            return None

    def update_local_save_status(
        self, filepath: str, upload_status: str, dataset_url: Optional[str] = None
    ) -> bool:
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            data["upload_status"] = upload_status
            data["upload_url"] = dataset_url
            data["last_save_time"] = datetime.now().isoformat()
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to update local save status: %s", e)
            return False

    def save_and_upload_detached(self, repo_id: str) -> Optional[str]:
        local_path = self.save_trajectory_local(upload_status="pending")
        if not local_path:
            return None
        try:
            uploader_script = Path(__file__).parent / "session_uploader.py"
            subprocess.Popen(
                [sys.executable, str(uploader_script), "upload", local_path, repo_id],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to spawn upload subprocess: %s", e)
        return local_path

    @staticmethod
    def retry_failed_uploads_detached(
        directory: str = "session_logs", repo_id: Optional[str] = None
    ) -> None:
        if not repo_id:
            return
        try:
            uploader_script = Path(__file__).parent / "session_uploader.py"
            subprocess.Popen(
                [sys.executable, str(uploader_script), "retry", directory, repo_id],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to spawn retry subprocess: %s", e)
