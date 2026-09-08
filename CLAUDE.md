# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Delegation (hard rule)

An orchestrating model must never write code in this repository — no edits, no new files, no patches. All code is written by delegated Sonnet 5 subagents.

The orchestrator's job is to read, investigate, design, brief the subagent with exact files and line numbers, and verify the resulting diff. Writing prose files (this one, docs, commit messages) is not code and stays with the orchestrator.

## Commands

```sh
uv sync                              # install dependencies
uv run -m marc                       # run Marc
uv run ruff check marc               # lint
uv run ruff format marc              # format (line-length 79)
```

Debugging the TUI: `print` goes nowhere useful because Textual owns the terminal. Use `self.log(...)` / `textual.log(...)`, then run `uv run textual console` in one shell and `uv run textual run --dev marc` in another.

There is no test suite and no test tooling configured. Verification is manual: run the app.

Requires `.env` with `OPENAI_API_KEY`, `TAVILY_API_KEY`, `MEM0_API_KEY` (see `example.env`). `load_dotenv()` runs at import of `marc/__init__.py`, which is why `LongTermMemory.init()` is deferred to `MarcApp.on_mount` — the Mem0 client cannot be constructed before env vars load.

## Architecture

Marc is a Textual TUI wrapping a LangChain agent. Three screens, one agent, three memory tiers.

### Agent construction

`marc/agent/__init__.py` is the center of the project: a single `SYSTEM_PROMPT_TEMPLATE` (`string.Template`) plus every tool definition. `create_new_agent()` substitutes three runtime values into that template — user profile, short-term memory, and the discovered skill catalog — and builds a fresh `create_agent` per chat screen. Prompt behavior is edited here, not in config; the template's `$`-placeholders mean literal `$` in the prompt must be escaped.

Two models throughout: `gpt-5.6-terra` for the main agent, `gpt-5.6-luna` for everything else (memory summarization, chat naming, computer-use subagent).

### Memory tiers

`marc/agent/memory.py` implements three tiers with deliberately different access patterns:

- **`UserMemory`** (`USER.md`) — agent-writable via `write_user_memory`, injected into every prompt, 100-word cap enforced by prompt only.
- **`ShortTermMemory`** (`TODAY.md`) — injected, read-only to the agent. Written on `ChatScreen.on_unmount` by a two-step LLM pass: summarize the session, then reconcile that summary into the existing log by session ID.
- **`LongTermMemory`** (Mem0, `app_id="com.rehatsingh.marc"`) — not injected; reached only through the `search_long_term_memory` tool.

**Dreaming** is the consolidation step and it runs *before* the first chat opens, never mid-session — `MarcApp.on_mount` checks `should_dream()` (is `TODAY.md`'s mtime older than today's midnight?) and pushes `DreamModeScreen` if so. `dream()` pushes STM into Mem0, rebuilds `LTM_INDEX.md` from the newly-added memories, then deletes STM. The invariant being protected is that memory never shifts underneath an in-flight conversation.

Note: `LTM_INDEX.md` is currently written but never read back into the prompt.

### Runtime context and UI updates

`RuntimeContext` (cwd + task list) is passed to `agent.astream(context=...)` and mutated in place by the task tools. `ChatScreen.send_message_to_agent` detects those mutations by comparing `hash(self.session.context)` across stream chunks, so **`RuntimeContext` and `Task` must stay hashable and hash over their mutable fields** — both define `__hash__`/`__eq__` explicitly for this reason. Breaking that silently stops the task panel and overlay from updating.

### Work-mode overlay

The overlay is a **separate process**, not a Textual widget: Dear PyGui cannot share the terminal event loop. `WorkModeScreen` spawns `python -m marc.work` and talks to it over piped stdin/stdout with newline-delimited JSON. Message types live in `marc/work/messages.py` as a discriminated union — adding a new one requires updating the `OverlayMessageType` union, or it is silently dropped by `model_validate_json`.

The parent forwards reasoning blocks and task-list changes as they stream; `TurnFinishedMessage` resets the overlay face and indicator.

### Computer-use subagent

`marc/agent/computer/` is a self-contained agent with its own prompt and tools, invoked through the main agent's `use_computer` tool. It is stateless by design — a fresh agent per delegation — so the user only ever has one conversational relationship, with Marc.

Screenshots are downscaled to `TARGET_WIDTH = 1280` and the ratio stored in `ComputerContext.scale_factor`; `click_mouse` multiplies incoming coordinates by that factor and adds the monitor's `left`/`top` offset. The model works entirely in downscaled image space and must never pre-scale coordinates itself.

### Skills

`marc/agent/skills.py` implements the [agentskills.io](https://agentskills.io/specification) spec: `SKILL.md` with YAML frontmatter. Discovery reads global skills from `~/.agents/skills` and project skills from the *current working directory itself* (bare skill directories in the CWD, not a subdirectory) — project skills shadow global ones by name. The catalog is inlined into the system prompt; the agent then loads a skill by calling its own `read_file` tool on the listed path.

### Persistence

Chats are JSON files at `<user_data_dir>/chats/<uuid>+<sanitized_name>.json` (`platformdirs`, app "Marc", author "CodeBoi"). The filename embeds the name so `list_chats` can build the switcher without parsing message bodies. Chat names are LLM-generated after the first turn.

## Design context

Product thinking and the task tracker live in Notion, not the repo — the README is a public-facing subset. Marc's positioning is "the coworker who's read all the docs": a peer, not a service, whose actual product is *judgment* about when to act, when to stay quiet, and when to volunteer something unasked. When changing prompts or adding features, that framing is the thing to check against.

The `shell_command` tool runs with no permission prompt and no sandbox. This is a known gap, not an oversight.
