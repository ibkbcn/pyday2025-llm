import argparse
import json
import os
from pathlib import Path
from typing import Any
from typing import TypeAlias

from openai import OpenAI
from rich.console import Console

from pyday2025_llm.constants import MODEL_NAME
from pyday2025_llm.tools import ListFilesParams
from pyday2025_llm.tools import ListFilesToolDefinition
from pyday2025_llm.tools import ReadFileParams
from pyday2025_llm.tools import ReadFileToolDefinition
from pyday2025_llm.tools import list_files
from pyday2025_llm.tools import read_file

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

        self.SYSTEM_PROMPT = f"""You are a helpful assistant.

You have access to tools to list files.

You base path is set to: {self.base_path_abs}, all the function calls MUST use paths relative to this base path, never use absolute paths.
For example, to list the folder f{self.base_path / "some_folder"}, you must only pass "some_folder" as the folder argument.
"""

    # TASK: Add validation to call_tool

    def call_tool(self, tool_name: str, parameters: dict) -> Any:
        validated_params: Any
        if tool_name == "list_files":
            validated_params = ListFilesParams.model_validate(parameters)
            result = list_files(Path(validated_params.folder))
        elif tool_name == "read_file":
            validated_params = ReadFileParams.model_validate(parameters)
            result = read_file(Path(validated_params.file_path))
        else:
            return "Unknown tool"

        if result.status_code != 0:
            return f"Error: {result.error_message}"
        return result.output

    def one_turn(self) -> Any:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.conversation_history,  # type: ignore
            tools=[{"type": "function", "function": t} for t in self.tools],  # type: ignore
            # TODO: add "tools" parameter
        )
        return response

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

        current_loop = 0

        while current_loop < self.max_loops:
            current_loop += 1

            response = self.one_turn()
            message = response.choices[0].message
            self.conversation_history.append(message.model_dump())

            # Print assistant text response if present
            if message.content:
                self.console.print(f"\n[yellow]Assistant[/yellow]: {message.content}")

            # Check if the model wants to call tools
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    self.console.print(
                        f"\n    [green]tool[/green]: {name}({json.dumps(arguments)})"
                    )

                    result = self.call_tool(name, arguments)
                    truncated = result[:200] + "..." if len(result) > 200 else result
                    self.console.print(f"        [dim]result[/dim]: {truncated}")

                    # Append tool result to conversation
                    self.conversation_history.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result,
                        }
                    )
            else:
                # No tool calls, model has finished this turn
                return message.content

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
        tools=[ListFilesToolDefinition, ReadFileToolDefinition],
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
