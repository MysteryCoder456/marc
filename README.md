# Marc

Marc is a cross-platform AI agent built on LangChain and LangGraph. The pitch: an agent that treats you like a professional.

> 🔨 **This project is an active WIP!** Follow development on [YouTube](https://www.youtube.com/@codeboi456) and [X](https://x.com/codeboi456_).

## Philosophy

Most assistants are tuned to *feel* helpful. Marc is tuned to have good judgment — about when to act, when to hold off, and when to bring up something you didn't ask about. The mental model I keep coming back to is the coworker who has read all the docs and knows where everything is.

What this means in practice:

- No "Great question!" padding. Short answers, real work in between.
- If Marc knows something relevant that you didn't ask about, it tells you.
- It assumes you know your own domain and doesn't over-explain.
- It's frugal with its context window: reads what it needs, doesn't redo work it already did, keeps answers brief.

None of this comes from a personality gimmick or a tone setting. It's how the agent is built to make decisions.

## What it can do

### 🧠 Memory
Marc remembers things across sessions, in three layers:

- **User profile** (`USER.md`) — a paragraph of stable facts about you, present in every session.
- **Short-term memory** (`TODAY.md`) — a summarized, running log of every interaction with Marc, injected into every new session.
- **Long-term memory** ([Mem0](https://mem0.ai)) — durable memory that Marc searches when older context becomes relevant.

### 🌙 Dreaming
Once a day, before the first chat opens, Marc consolidates yesterday's short-term memory into long-term memory and starts a fresh daily log. You'll see a "Marc is Dreaming" screen while it happens. The point of doing it this way is that memory never shifts underneath you mid-conversation.

### 🖥️ Computer use
For GUI tasks, Marc hands off to an internal subagent that can look at your screen and act on it. You give it a goal in plain language and get a plain-language report back. You never talk to subagents directly — the whole point is that you have one relationship, with Marc.

### 🔎 Web research
Marc uses the full [Tavily](https://tavily.com) toolset (Search, Extract, Map, Crawl), starting with the cheapest tool that can answer and escalating only when needed. All web access goes through these tools. 

### ⚡ Shell access
Marc can run shell commands for local work. Fair warning: this is the roughest part of the project right now. Commands run without asking you first, and there's no sandbox. I'd recommend going in and disabling shell tools in `marc/agent/__init__.py` if you're going to try Marc until permission interrupts are implemented.

### 🪟 Work-mode overlay
A small on-screen overlay that sits alongside your work while Marc is active: Marc's face as he works, his reasoning streaming live, and a panel showing the current session's tasks. It checks in with you periodically as it works.

## Not done yet

- **Model selection** — models are fixed for now; picking your own is coming.
- **Shell hardening** — see above. No permission prompts, no sandboxing yet.
- **Observation** — the idea is that Marc can watch your work sessions and learn your routines well enough to notice patterns you didn't point out. Still in design; I'm working through the trade-off between always-on observation (more signal) and opt-in, session-scoped observation (better for privacy).
- **Recall-driven proactivity** — Marc surfacing your own past work when it's relevant again, like a data structure you designed weeks ago and forgot about.
- **Memory provenance** — tracking where memories came from, how confident Marc is in them, and when they were last confirmed.

## Stack

- LangChain + LangGraph for orchestration
- OpenAI models only for now — GPT-5.6-terra for the main agent, GPT-5.6-luna for everything else. One provider keeps things simple to set up and test while the project is young; model and provider selection is on the roadmap.
- Mem0 for long-term memory
- Textual for the TUI
- Tavily for web access
- Dear PyGui for the overlay
