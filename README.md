<p align="center">
  <img src="frontend/public/smolagents.webp" alt="smolagents logo" width="160" />
</p>

# ML Intern

An ML intern that autonomously researches, writes, and ships good-quality ML-related code using the Hugging Face ecosystem — with deep access to docs, papers, datasets, and cloud compute.

Built on the [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python) (the turn loop, tool execution, and auto-compaction all come from the SDK; this project provides the HF-specific tools, prompts, and session plumbing).

## Quick Start

### Installation

```bash
git clone git@github.com:huggingface/ml-intern.git
cd ml-intern
uv sync
uv tool install -e .
```

#### That's it. Now `ml-intern` works from any directory:

```bash
ml-intern
```

Create a `.env` file in the project root (or export these in your shell):

```bash
ANTHROPIC_API_KEY=<your-anthropic-api-key>   # required — Claude Agent SDK
HF_TOKEN=<your-hugging-face-token>           # required — HF tool calls, job submission
GITHUB_TOKEN=<github-personal-access-token>  # optional — higher GitHub rate limits
```

If no `HF_TOKEN` is set, the CLI will prompt you to paste one on first launch. To get a `GITHUB_TOKEN` follow the tutorial [here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-fine-grained-personal-access-token).

### Usage

**Interactive mode** (start a chat session):

```bash
ml-intern
```

**Headless mode** (single prompt, auto-approve):

```bash
ml-intern "fine-tune llama on my dataset"
```

**Options:**

```bash
ml-intern --model claude-opus-4-7 "your prompt"
ml-intern --max-iterations 100 "your prompt"
ml-intern --no-stream "your prompt"
```

Supported models: `claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5-20251001`. The Claude Agent SDK routes through the bundled Claude Code CLI; HF Router models (Kimi, GLM, MiniMax) are not supported on this code path.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                        User / CLI                        │
└───────────┬────────────────────────────────────┬─────────┘
            │ Operations                         │ Events
            ↓                                    ↑
     submission_queue                      event_queue
            │                                    │
            ↓                                    │
┌──────────────────────────────────────────────┐ │
│  submission_loop (agent_loop.py)             │ │
│    dispatches OpType → SdkRunner             │ │
└───────────────────┬──────────────────────────┘ │
                    ↓                            │
┌──────────────────────────────────────────────┐ │
│  SdkRunner                                   │ │
│    ClaudeSDKClient  ─┐                       │ │
│                      │ streams messages      │ │
│                      ▼                       │ │
│    translator: AssistantMessage/ToolUseBlock/│─┤
│                UserMessage(ToolResultBlock)/ │ │
│                ResultMessage → Event         │ │
└──────────────────────────────────────────────┘ │
                    │                            │
  ┌─────────────────┼─────────────────┐          │
  ↓                 ↓                 ↓          │
┌────────────┐ ┌──────────┐ ┌──────────────────┐ │
│ hf-tools   │ │ HF MCP   │ │ PreToolUse hooks │─┘
│ (in-proc   │ │ (remote) │ │  • approval gate │
│  @tool)    │ │          │ │  • doom-loop     │
└────────────┘ └──────────┘ └──────────────────┘
```

**What the Claude Agent SDK owns:** conversation history, turn loop, streaming, auto-compaction, retries, tool execution, permission prompts.

**What ML Intern adds:** HF-specific tool suite (papers, jobs, sandbox, repo management, dataset inspection, docs search, GitHub, research sub-agent), the ML-engineering system prompt (`prompts/system_prompt_v3.yaml`), HF OAuth + session management for the web UI, trajectory logging to an HF dataset, and approval/doom-loop hooks.

## Events

The agent emits the following events via `event_queue`:

- `processing` - Starting to process user input
- `ready` - Agent is ready for input
- `assistant_chunk` - Streaming token chunk
- `assistant_message` - Complete LLM response text
- `assistant_stream_end` - Token stream finished
- `tool_call` - Tool being called with arguments
- `tool_output` - Tool execution result
- `tool_log` - Informational tool log message
- `tool_state_change` - Tool execution state transition
- `approval_required` - Requesting user approval for sensitive operations
- `turn_complete` - Agent finished processing
- `error` - Error occurred during processing
- `interrupted` - Agent was interrupted
- `compacted` - Context was compacted
- `undo_complete` - Undo operation completed
- `shutdown` - Agent shutting down

## Development

### Adding Built-in Tools

Edit `agent/core/tools.py`:

```python
def create_builtin_tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="your_tool",
            description="What your tool does",
            parameters={
                "type": "object",
                "properties": {
                    "param": {"type": "string", "description": "Parameter description"}
                },
                "required": ["param"]
            },
            handler=your_async_handler
        ),
        # ... existing tools
    ]
```

### Adding MCP Servers

Edit `configs/main_agent_config.json`:

```json
{
  "model_name": "anthropic/claude-sonnet-4-5-20250929",
  "mcpServers": {
    "your-server-name": {
      "transport": "http",
      "url": "https://example.com/mcp",
      "headers": {
        "Authorization": "Bearer ${YOUR_TOKEN}"
      }
    }
  }
}
```

Note: Environment variables like `${YOUR_TOKEN}` are auto-substituted from `.env`.
