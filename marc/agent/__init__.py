import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AnyMessage, ContentBlock
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from .computer import create_computer_use_agent


@dataclass
class RuntimeContext:
    cwd: Path


SYSTEM_PROMPT = """# Marc System Prompt

You are Marc, a practical desktop assistant. Help the user complete tasks on
their computer with clear judgment, concise communication, and the available
tools.

## Operating Principles

- Be direct and useful. Prefer doing the work over describing how to do it.
- Use the smallest reliable tool for the job: read files before editing them,
  list directories before assuming paths, and run non-interactive shell commands
  when they are the right fit.
- Preserve user work. Do not overwrite files, run destructive shell commands,
  install software, change system settings, or perform irreversible actions
  unless the user explicitly asked for that outcome.
- If a request is ambiguous and a wrong assumption could damage data, spend
  money, publish private information, or change external state, ask one focused
  question before acting.
- Be honest about uncertainty, failures, and partial progress. Do not claim a
  task is complete until the relevant result has been verified.
- Keep final responses brief: state what changed or what you found, mention any
  blockers, and include the most useful next step only when needed.

## Files and Shell

- Interpret relative paths from the current working directory.
- Prefer structured file tools for reading and writing known files.
- Use shell commands for discovery, tests, build steps, and operations that are
  easier or safer through the shell.
- For long-running or interactive commands, use a non-interactive form whenever
  possible and explain the limitation if the command times out.

## Computer-Use Delegation

Use `use_computer` only for tasks that require interacting with the
graphical desktop: clicking, typing into apps, navigating windows, reading
visual UI state, or verifying something that only appears on screen.

Before delegating, gather any useful non-visual context yourself. Send the
computer-use subagent a compact, self-contained instruction that includes:

- the exact goal and success criteria;
- the app, window, website, or visible UI target if known;
- relevant text, paths, credentials placeholders, or constraints;
- actions to avoid, especially destructive, privacy-sensitive, or paid actions;
- what evidence you need back, such as visible confirmation text.

After the subagent returns, use its result to answer the user. If the subagent
reports uncertainty, a blocker, or a need for confirmation, surface that clearly
instead of guessing.
"""


@tool
def get_system_info() -> tuple[str]:
    """
    Get information about the user's system like OS, processor, etc.

    Returns:
        A tuple containing system information.
    """

    info = platform.uname()
    return tuple(info)  # pyright: ignore[reportReturnType]


@tool
def read_file(path: Path, runtime: ToolRuntime[RuntimeContext]) -> str:
    """
    Read the contents of the file at the specified path.

    Args:
        path:
            Path to the file to be read. By default, this is interpreted as a
            relative path to the current working directory. Use the
            appropriate root path prefix to specify an absolute path (`C:\\` on
            Windows, `/` on Unix-based systems).

    Returns:
        Contents of the file as a string
    """

    if not path.is_absolute():
        path = runtime.context.cwd / path

    with open(path, "r") as f:
        return f.read()


@tool
def write_file(
    path: Path, contents: str, runtime: ToolRuntime[RuntimeContext]
):
    """
    Write to the file at the specified path. This will overwrite any existing
    data in the file.

    Args:
        path:
            Path to the file to be written to. By default, this is interpreted
            as a relative path to the current working directory. Use the
            appropriate root path prefix to specify an absolute path (`C:\\` on
            Windows, `/` on Unix-based systems).
        contents:
            New contents to be written to the file.
    """

    if not path.is_absolute():
        path = runtime.context.cwd / path

    with open(path, "w") as f:
        _ = f.write(contents)


@tool
def list_dir(path: Path, runtime: ToolRuntime[RuntimeContext]) -> list[str]:
    """
    List the files in directory at the specified path.

    Args:
        path:
            Path to the directory. By default, this is interpreted as a
            relative path to the current working directory. Use the
            appropriate root path prefix to specify an absolute path (`C:\\` on
            Windows, `/` on Unix-based systems).

    Return:
        A list of file and directory names in the specified directory
    """

    if not path.is_absolute():
        path = runtime.context.cwd / path

    return os.listdir(path)


@tool
def shell_command(
    cmd: str, runtime: ToolRuntime[RuntimeContext]
) -> str | tuple[str, str]:
    """
    Execute a shell command on the user's system at the current working
    directory. A timeout of 30s exists for any command executed.

    Always run non-interactive versions of commands whenever possible.

    Args:
        cmd: The shell command to execute.

    Returns:
        Output of the shell command as a tuple containing `(stdout, stderr)`,
        or the string `TIMEOUT` if the command timed out.
    """

    # TODO: handle Windows case

    user_shell = os.getenv("SHELL") or "/bin/bash"

    try:
        result = subprocess.run(
            [user_shell, "-i", "-c", cmd],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=runtime.context.cwd,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        return result.stdout, result.stderr

    except subprocess.TimeoutExpired:
        return "TIMEOUT"


@tool
def use_computer(query: str) -> list[ContentBlock]:
    """
    Create an ephemeral subagent to handle computer-use tasks. Use relevant
    context to describe what you want the agent to do on the user's computer.

    Args:
        query:
            A natural language query describing the task you want the agent
            to perform.
    """

    computer_agent = create_computer_use_agent()
    result = computer_agent.invoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    final_msg: AnyMessage = result["messages"][-1]
    return final_msg.content_blocks


def create_new_agent():
    memory = InMemorySaver()

    model = ChatOpenAI(
        model="gpt-5.4-mini",
        reasoning={"effort": "medium", "summary": "concise"},
    )
    agent = create_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=memory,
        context_schema=RuntimeContext,
        tools=[  # TODO: tool to change CWD
            get_system_info,
            read_file,
            write_file,
            list_dir,
            shell_command,
            use_computer,
        ],
        # middleware=[AnthropicPromptCachingMiddleware()],
    )
    return agent
