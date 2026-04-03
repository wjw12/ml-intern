"""
Terminal display utilities — rich-powered CLI formatting.
"""

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.theme import Theme

_THEME = Theme({
    "tool.name": "bold cyan",
    "tool.args": "dim",
    "tool.ok": "dim green",
    "tool.fail": "dim red",
    "info": "dim",
    "muted": "dim",
})

_console = Console(theme=_THEME, highlight=False)

# Indent prefix for all agent output (aligns under the `>` prompt)
_I = "  "


def get_console() -> Console:
    return _console


# ── Banner ─────────────────────────────────────────────────────────────

def print_banner() -> None:
    Y = "\033[38;2;255;200;50m"  # HF yellow
    D = "\033[38;2;180;140;40m"  # dimmer gold for accents
    R = "\033[0m"
    art = f"""
{_I}{Y} _  _                _             ___                _                _   {R}
{_I}{Y}| || |_  _ __ _ __ _(_)_ _  __ _  | __|_ _ __ ___    /_\\  __ _ ___ _ _| |_ {R}
{_I}{Y}| __ | || / _` / _` | | ' \\/ _` | | _/ _` / _/ -_)  / _ \\/ _` / -_) ' \\  _|{R}
{_I}{Y}|_||_|\\_,_\\__, \\__, |_|_||_\\__, | |_|\\__,_\\__\\___| /_/ \\_\\__, \\___|_||_\\__|{R}
{_I}{D}          |___/|___/       |___/                         |___/             {R}
"""
    _console.print(art, highlight=False)
    _console.print(f"{_I}[dim]🤗 /help for commands · /quit to exit[/dim]\n")


# ── Init progress ──────────────────────────────────────────────────────

def print_init_done() -> None:
    _console.print(f"{_I}[dim]Ready.[/dim]\n")


# ── Tool calls ─────────────────────────────────────────────────────────

def print_tool_call(tool_name: str, args_preview: str) -> None:
    _console.print(f"{_I}[tool.name]▸ {tool_name}[/tool.name]  [tool.args]{args_preview}[/tool.args]")


def print_tool_output(output: str, success: bool, truncate: bool = True) -> None:
    if truncate:
        output = _truncate(output, max_lines=10)
    style = "tool.ok" if success else "tool.fail"
    # Indent each line of tool output
    indented = "\n".join(f"{_I}  {line}" for line in output.split("\n"))
    _console.print(f"[{style}]{indented}[/{style}]")


class SubAgentDisplay:
    """Live-updating display: header with stats (ticks every second) + rolling 2-line tool calls."""

    _MAX_VISIBLE = 2

    def __init__(self):
        self._calls: list[str] = []
        self._tool_count = 0
        self._token_count = 0
        self._start_time: float | None = None
        self._lines_on_screen = 0
        self._ticker_task = None

    def start(self) -> None:
        """Begin the display with a 1-second ticker."""
        import asyncio
        import time
        self._calls = []
        self._tool_count = 0
        self._token_count = 0
        self._start_time = time.monotonic()
        self._redraw()
        self._ticker_task = asyncio.ensure_future(self._tick())

    def set_tokens(self, tokens: int) -> None:
        self._token_count = tokens
        # no redraw — ticker handles it

    def set_tool_count(self, count: int) -> None:
        self._tool_count = count
        # no redraw — ticker handles it

    def add_call(self, tool_desc: str) -> None:
        self._calls.append(tool_desc)
        self._redraw()

    def clear(self) -> None:
        if self._ticker_task:
            self._ticker_task.cancel()
            self._ticker_task = None
        self._erase()
        self._lines_on_screen = 0
        self._calls = []
        self._start_time = None

    async def _tick(self) -> None:
        import asyncio
        try:
            while True:
                await asyncio.sleep(1.0)
                self._redraw()
        except asyncio.CancelledError:
            pass

    def _format_stats(self) -> str:
        import time
        if self._start_time is None:
            return ""
        elapsed = time.monotonic() - self._start_time
        if elapsed < 60:
            time_str = f"{elapsed:.0f}s"
        else:
            time_str = f"{elapsed / 60:.0f}m {elapsed % 60:.0f}s"
        if self._token_count >= 1000:
            tok_str = f"{self._token_count / 1000:.1f}k"
        else:
            tok_str = str(self._token_count)
        return f"{self._tool_count} tool uses · {tok_str} tokens · {time_str}"

    def _erase(self) -> None:
        if self._lines_on_screen > 0:
            f = _console.file
            for _ in range(self._lines_on_screen):
                f.write("\033[A\033[K")
            f.flush()

    def _redraw(self) -> None:
        f = _console.file
        self._erase()
        lines = []
        # Header: ▸ research (stats)
        stats = self._format_stats()
        header = f"{_I}\033[1;36m▸ research\033[0m"
        if stats:
            header += f"  \033[2m({stats})\033[0m"
        lines.append(header)
        # Last 2 tool calls, gray
        visible = self._calls[-self._MAX_VISIBLE:]
        for desc in visible:
            lines.append(f"{_I}  \033[2m{desc}\033[0m")
        for line in lines:
            f.write(line + "\n")
        f.flush()
        self._lines_on_screen = len(lines)


_subagent_display = SubAgentDisplay()


def print_tool_log(tool: str, log: str) -> None:
    """Handle tool log events — sub-agent calls get the rolling display."""
    if tool == "research":
        if log == "Starting research sub-agent...":
            _subagent_display.start()
        elif log == "Research complete.":
            _subagent_display.clear()
        elif log.startswith("tokens:"):
            _subagent_display.set_tokens(int(log[7:]))
        elif log.startswith("tools:"):
            _subagent_display.set_tool_count(int(log[6:]))
        else:
            _subagent_display.add_call(log)
    else:
        _console.print(f"{_I}[dim]{tool}: {log}[/dim]")


# ── Messages ───────────────────────────────────────────────────────────

def print_markdown(text: str) -> None:
    from rich.padding import Padding
    _console.print()
    _console.print(Padding(Markdown(text), (0, 0, 0, 2)))


def print_error(message: str) -> None:
    _console.print(f"\n{_I}[bold red]Error:[/bold red] {message}")


def print_turn_complete() -> None:
    pass  # no separator — clean output


def print_interrupted() -> None:
    _console.print(f"\n{_I}[dim italic]interrupted[/dim italic]")


def print_compacted(old_tokens: int, new_tokens: int) -> None:
    _console.print(f"{_I}[dim]context compacted: {old_tokens:,} → {new_tokens:,} tokens[/dim]")


# ── Approval ───────────────────────────────────────────────────────────

def print_approval_header(count: int) -> None:
    label = f"Approval required — {count} item{'s' if count != 1 else ''}"
    _console.print()
    _console.print(f"{_I}", Panel(f"[bold yellow]{label}[/bold yellow]", border_style="yellow", expand=False))


def print_approval_item(index: int, total: int, tool_name: str, operation: str) -> None:
    _console.print(f"\n{_I}[bold]\\[{index}/{total}][/bold]  [tool.name]{tool_name}[/tool.name]  {operation}")


def print_yolo_approve(count: int) -> None:
    _console.print(f"{_I}[bold yellow]yolo →[/bold yellow] auto-approved {count} item(s)")


# ── Help ───────────────────────────────────────────────────────────────

HELP_TEXT = f"""\
{_I}[bold]Commands[/bold]
{_I}  [cyan]/help[/cyan]            Show this help
{_I}  [cyan]/undo[/cyan]            Undo last turn
{_I}  [cyan]/compact[/cyan]         Compact context window
{_I}  [cyan]/model[/cyan] [id]      Show available models or switch
{_I}  [cyan]/yolo[/cyan]            Toggle auto-approve mode
{_I}  [cyan]/status[/cyan]          Current model & turn count
{_I}  [cyan]/quit[/cyan]            Exit"""


def print_help() -> None:
    _console.print()
    _console.print(HELP_TEXT)
    _console.print()


# ── Plan display ───────────────────────────────────────────────────────

def format_plan_display() -> str:
    """Format the current plan for display."""
    from agent.tools.plan_tool import get_current_plan

    plan = get_current_plan()
    if not plan:
        return ""

    completed = [t for t in plan if t["status"] == "completed"]
    in_progress = [t for t in plan if t["status"] == "in_progress"]
    pending = [t for t in plan if t["status"] == "pending"]

    lines = []
    for t in completed:
        lines.append(f"{_I}[green]✓[/green] [dim]{t['content']}[/dim]")
    for t in in_progress:
        lines.append(f"{_I}[yellow]▸[/yellow] {t['content']}")
    for t in pending:
        lines.append(f"{_I}[dim]○ {t['content']}[/dim]")

    summary = f"[dim]{len(completed)}/{len(plan)} done[/dim]"
    lines.append(f"{_I}{summary}")
    return "\n".join(lines)


def print_plan() -> None:
    plan_str = format_plan_display()
    if plan_str:
        _console.print(plan_str)


# ── Formatting for plan_tool output (used by plan_tool handler) ────────

def format_plan_tool_output(todos: list) -> str:
    if not todos:
        return "Plan is empty."

    lines = ["Plan updated:", ""]
    completed = [t for t in todos if t["status"] == "completed"]
    in_progress = [t for t in todos if t["status"] == "in_progress"]
    pending = [t for t in todos if t["status"] == "pending"]

    for t in completed:
        lines.append(f"  [x] {t['id']}. {t['content']}")
    for t in in_progress:
        lines.append(f"  [~] {t['id']}. {t['content']}")
    for t in pending:
        lines.append(f"  [ ] {t['id']}. {t['content']}")

    lines.append(f"\n{len(completed)}/{len(todos)} done")
    return "\n".join(lines)


# ── Internal helpers ───────────────────────────────────────────────────

def _truncate(text: str, max_lines: int = 6) -> str:
    lines = text.split("\n")
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"
