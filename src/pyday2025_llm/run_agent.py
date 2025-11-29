import argparse
import os
from pathlib import Path
from typing import Any
from typing import TypeAlias

from openai import OpenAI
from rich.console import Console

from pyday2025_llm.constants import MODEL_NAME
from pyday2025_llm.tools import ListFilesToolDefinition

# from pyday2025_llm.tools import list_files

Conversation: TypeAlias = list[dict]

"""
TASKS:

- Call single LLM turn
- Add tools to Agent class
- Check how to send tool info to OpenAI chat completions
- Improve SYSTEM_PROMPT
- Add call_tool method
- Add --debug flag to CLI and implement conversation saving
- Add --user-input CLI flag

"""


def get_openai_key() -> str:
    """
    Retrieve the OpenAI API key from environment or file.

    Returns:
        The OpenAI API key as a string.
    """

    api_key_env = os.getenv("OPENAI_API_KEY")
    if api_key_env:
        return api_key_env
    api_key_file = Path.cwd() / ".openai_api_key"
    if api_key_file.exists():
        file_contents = api_key_file.read_text().strip()
        if file_contents:
            return file_contents
    raise ValueError("OpenAI API key not found in environment or .openai_api_key")


class Agent:
    def __init__(
        self,
        client: OpenAI,
        max_loops: int = 50,
        base_path: Path | None = None,
        tools: list[dict] | None = None,
    ):
        self.client = client
        self.model_name = MODEL_NAME
        self.max_loops = max_loops
        self.conversation_history: Conversation = []
        self.base_path = base_path or Path("data")
        self.base_path_abs = self.base_path.resolve()
        self.console = Console()
        self.tools = tools or []

        self.SYSTEM_PROMPT = """You are a helpful assistant.
""".strip()

    # TASK: Add call_tool
    # TASK: Add validation to call_tool

    def one_turn(self) -> Any:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.conversation_history,  # type: ignore
            tools=[{"type": "function", "function": t} for t in self.tools],  # type: ignore
            # TODO: add "tools" parameter
        )
        return response

    # TASK: Add conversation saving with --debug flag

    def run(self, user_input: str | None = None):
        # Initialize conversation with system prompt if starting fresh
        if not self.conversation_history:
            self.conversation_history = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
            ]

        # Add user input if provided
        if user_input:
            self.conversation_history.append({"role": "user", "content": user_input})
            # self.save_conversation()

        response = self.one_turn()
        message = response.choices[0].message
        self.conversation_history.append(message.model_dump())
        return self.conversation_history[-1].get("content", "")


def main() -> int:
    """Main CLI function."""

    args = parse_args()
    openai_api_key = get_openai_key()
    client = OpenAI(api_key=openai_api_key)
    max_loops = args.max_loops
    base_path = Path(args.base_path)

    agent = Agent(
        client=client,
        max_loops=max_loops,
        base_path=base_path,
        tools=[ListFilesToolDefinition],
    )

    console = Console()
    console.print("Chat with the agent (use 'ctrl-c' to quit)")

    user_input = console.input("\n[blue]You[/blue]: ")
    if not user_input.strip():
        console.print("No input provided.")
    return agent.run(user_input)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run agent search loop to find best match for a query."
    )
    parser.add_argument(
        "--max-loops",
        type=int,
        default=50,
        help="Maximum number of search iterations. Default: %(default)s",
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default="data",
        help="Base path for file operations. Default: %(default)s",
    )
    parser.add_argument(
        "--user-input",
        type=str,
        default=None,
        help="User input to start with. If not provided, will prompt interactively.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
