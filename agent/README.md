# Agent

The agent runs on the **[Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python)**. The SDK owns the LLM turn mechanics (streaming, tool execution, auto-compaction); this package provides the ML-intern specific tools, prompts, and session plumbing.

## Layout

| Module | Purpose |
|---|---|
| `core/session.py` | Per-session state: event queue, cancellation flag, sandbox handle, trajectory logging, HF token, config. No conversation history — the SDK owns that. |
| `core/sdk_runner.py` | Thin adapter that owns a `ClaudeSDKClient`, pumps `receive_response()` messages out as UI events (`assistant_message`, `tool_call`, `tool_output`, `turn_complete`, `compacted`). |
| `core/sdk_tools.py` | Wraps every ML-intern tool handler as `@tool` functions inside an in-process `create_sdk_mcp_server("hf-tools", ...)`. Each session builds its own server so handlers can close over `Session` (for HF token + sandbox access). |
| `core/sdk_hooks.py` | `PreToolUse` hook: approval gating (replaces the old `_needs_approval`) and doom-loop detection. `ApprovalManager` bridges the hook to the existing `EXEC_APPROVAL` op via an `asyncio.Future`. |
| `core/sdk_options.py` | Builds the `ClaudeAgentOptions` for a session: system prompt, cwd, MCP servers, allow/disallow lists, hooks, permission mode, `max_turns`. |
| `core/agent_loop.py` | `submission_loop` + `process_submission` — dispatches `USER_INPUT`/`EXEC_APPROVAL`/`INTERRUPT`/`COMPACT`/`SHUTDOWN` ops onto the runner. |
| `tools/*.py` | The actual ML-intern tool implementations (papers, jobs, sandbox, repo mgmt, docs search, GitHub, dataset inspection, plan, research sub-agent). |
| `prompts/system_prompt_v3.yaml` | Active system prompt. Rendered with Jinja2 and passed to `ClaudeAgentOptions(system_prompt=...)`. |

## How a user turn flows

```
UI submits OpType.USER_INPUT
         │
         ▼
submission_loop → process_submission
         │
         ▼
SdkRunner.run_turn(text)
         │
         ├─ ClaudeSDKClient.query(text)
         ▼
async for msg in client.receive_response():
    AssistantMessage(TextBlock)   → event: assistant_message
    AssistantMessage(ToolUseBlock)→ event: tool_call
    UserMessage(ToolResultBlock)  → event: tool_output
    ResultMessage                 → event: turn_complete
```

Approval flow:
1. Claude issues a tool call.
2. `PreToolUse` hook runs `_needs_approval` → if yes, returns `permissionDecision: "ask"`.
3. The hook suspends on a `Future` keyed by `tool_use_id`; emits `approval_required`.
4. UI returns `EXEC_APPROVAL` with a decision; `ApprovalManager.resolve_all` wakes the hook.
5. Hook returns `allow`/`deny` and Claude proceeds.

Doom-loop detection: the same `PreToolUse` hook tracks recent tool signatures (name + args hash) per session; if it sees the same call three times in a row or a repeating `[A,B,A,B]` pattern, it returns `additionalContext` that tells Claude to change strategy.

Compaction: handled by the Claude Code CLI. `PreCompact` is available if you need custom summarization instructions — register it via `ClaudeAgentOptions(hooks={"PreCompact": [...]})`.
