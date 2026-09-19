# Marc

Marc is a cross-platform AI agent built on LangChain. The pitch: an agent that treats you like a professional.

## Philosophy

I started Marc because I wanted an assistant that could do useful work without constantly getting in the way. It should know when to act, when to ask, and when something from an earlier conversation is worth bringing up. The closest comparison is a coworker who has read the docs, remembers what you've worked on, and doesn't need every bit of context explained again.

That leads to a few basic rules:

- Skip canned enthusiasm and get to the answer.
- Bring up relevant context, even when the user didn't explicitly ask for it.
- Assume the user understands their own field instead of explaining everything from first principles.
- Read only what's needed and avoid doing the same work twice.

These are partly personality settings, but more so capabilities as well. They shape how Marc decides what to read, what to remember, and what to do.

## Capabilities

### 🧠 Memory
Marc remembers things across sessions, in three layers:

- **User profile** (`USER.md`) — a paragraph of stable facts about you, present in every session.
- **Short-term memory** (`TODAY.md`) — a summarized, running log of every interaction with Marc you've had today, injected into every new session.
- **Long-term memory** ([Mem0](https://mem0.ai)) — durable memory that Marc searches when older context becomes relevant.

### 🌙 Dreaming
Before the first chat of the day opens, Marc consolidates the previous day's short-term memory into long-term memory and starts a fresh daily log. A "Marc is Dreaming" screen is shown while this runs.

### 🖥️ Computer use
For GUI tasks, Marc hands work off to an internal subagent that can view the screen and interact with it. It accepts a goal in plain language and returns a report to the main chat.

### 🔎 Web research
Marc uses the full [Tavily](https://tavily.com) toolset: Search, Extract, Map, and Crawl. All web access goes through these tools.

### ⚡ Shell access
Marc can run shell commands for local work. Fair warning: this is the roughest part of the project right now. Commands run without asking you first, and there's no sandbox. I'd recommend going in and disabling shell tools in `marc/agent/__init__.py` if you're going to try Marc until permission interrupts are implemented.

### 🪟 Work-mode overlay
A small on-screen overlay appears while Marc is active. It shows Marc's face, live reasoning output, and the current session's tasks. It can also prompt the user for input while work is in progress.

### 👁️ Observation
Observation is an opt-in mode toggled with `Ctrl+O`. While it is active, Marc captures changes across the user's screens, analyzes the current activity, searches memory for relevant context, and sends useful findings to the main agent. The observation session ends when the mode is toggled off.

## Installation & Usage

Marc uses [uv](https://docs.astral.sh/uv/) as its package manager. It is also the recommended tool to run Marc while plans to make it installable via `pip` are underway.

Three primary APIs are required to use Marc: [OpenAI](https://platform.openai.com/) (or any other LLM provider), [Tavily](https://www.tavily.com/), and [Mem0](https://www.tavily.com/). Tavily and Mem0 have generous free tiers (at least for my usage). While you will have to bring your own API key for OpenAI's ChatGPT models, they do have a [data sharing tier](https://help.openai.com/en/articles/10306912-sharing-feedback-evaluation-and-fine-tuning-data-and-api-inputs-and-outputs-with-openai#h_f2f71332e6) which grants you a certain amount of free tokens daily.

1. Install uv by following [these](https://docs.astral.sh/uv/getting-started/installation/) instructions.
2. Install Marc's dependencies by running the command `uv sync` in your shell.
3. Create a copy of `example.env` called `.env`, and fill in your API keys.
4. Run `uv run -m marc` in your shell to start Marc.
5. (Optional) Run `uv run textual console`, then start `uv run textual run --dev marc` in another shell. This will allow you to see print statements and other debugging information.

## Stack

- LangChain + LangGraph for orchestration
- OpenAI models only for now — GPT-5.6-terra for the main agent, GPT-5.6-luna for everything else. One provider keeps things simple to set up and test while the project is young; model and provider selection is on the roadmap.
- Mem0 for long-term memory
- Textual for the TUI
- Tavily for web access
- Dear PyGui for the overlay
