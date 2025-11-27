import argparse
import json
import os
from pathlib import Path
from typing import Any
from typing import TypeAlias

from openai import OpenAI
from rich.console import Console

from pyday2025_llm.constants import MODEL_NAME
from pyday2025_llm.tools import GrepPatternTool
from pyday2025_llm.tools import ListFilesTool
from pyday2025_llm.tools import ReadFileTool
from pyday2025_llm.tools import grep_pattern
from pyday2025_llm.tools import list_files
from pyday2025_llm.tools import read_file

Conversation: TypeAlias = list[dict]


class Agent:
    def __init__(
        self,
        client: OpenAI,
        model_name: str = MODEL_NAME,
        max_loops: int = 50,
        tools: list[dict] | None = None,
        base_path: Path | None = None,
        debug: bool = False,
    ):
        self.client = client
        self.model_name = model_name
        self.max_loops = max_loops
        self.conversation_history: Conversation = []
        self.tools = tools or []
        self.base_path = base_path or Path("data")
        self.base_path_abs = self.base_path.resolve()
        self.debug = debug
        self.console = Console()

        self.SYSTEM_PROMPT = f"""You are a helpful assistant.

You have access to tools to list files, read files, and search for patterns.

You base path is set to: {self.base_path_abs}, all the function calls MUST use paths relative to this base path, never use absolute paths.
For example, to list the folder f{self.base_path / "some_folder"}, you must only pass "some_folder" as the folder argument.
"""

    def call_tool(self, name: str, arguments: dict) -> str:
        """Execute a tool and return its result."""
        if name == "list_files":
            folder = self.base_path / arguments["folder"]
            result = list_files(folder)
        elif name == "read_file":
            file_path = self.base_path / arguments["file_path"]
            result = read_file(
                file_path,
                arguments.get("line_start"),
                arguments.get("line_end"),
            )
        elif name == "grep_pattern":
            base_path = self.base_path / arguments["base_path"]
            result = grep_pattern(base_path, arguments["pattern"])
        else:
            return f"Unknown tool: {name}"

        if result.status_code != 0:
            return f"Error: {result.error_message}"
        return result.output

    def one_turn(self) -> Any:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.conversation_history,  # type: ignore
            tools=[{"type": "function", "function": t} for t in self.tools],  # type: ignore
        )
        return response

    def save_conversation(self):
        """Save conversation to conversation.json if debug mode is enabled."""
        if self.debug:
            with open("conversation.json", "w") as f:
                json.dump(self.conversation_history, f, indent=2)

    def run(self, user_input: str | None = None):
        # Initialize conversation with system prompt if starting fresh
        if not self.conversation_history:
            self.conversation_history = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
            ]

        # Add user input if provided
        if user_input:
            self.conversation_history.append({"role": "user", "content": user_input})
            # self.console.print(f"\n[blue]You[/blue]: {user_input}")
            self.save_conversation()

        current_loop = 0

        while current_loop < self.max_loops:
            current_loop += 1

            response = self.one_turn()
            message = response.choices[0].message

            # Append assistant message to conversation
            self.conversation_history.append(message.model_dump())
            self.save_conversation()

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
                    self.save_conversation()
            else:
                # No tool calls, model has finished this turn
                return message.content

        print("Max loops reached, stopping.")
        return self.conversation_history[-1].get("content", "")


def main() -> int:
    """Main CLI function."""

    args = parse_args()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=openai_api_key)
    max_loops = args.max_loops
    model_name = args.model_name
    base_path = Path(args.base_path)
    debug = args.debug

    tools = [ListFilesTool, ReadFileTool, GrepPatternTool]

    agent = Agent(
        client=client,
        model_name=model_name,
        max_loops=max_loops,
        tools=tools,
        base_path=base_path,
        debug=debug,
    )

    console = Console()
    console.print("Chat with the agent (use 'ctrl-c' to quit)")

    # If user input provided via CLI, run once and exit
    if args.user_input:
        agent.run(args.user_input)
        return 0

    # Multi-turn conversation loop
    while True:
        try:
            user_input = console.input("\n[blue]You[/blue]: ")
            if not user_input.strip():
                continue
            agent.run(user_input)
        except (KeyboardInterrupt, EOFError):
            console.print("\nGoodbye!")
            break

    return 0


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
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help="OpenAI model name to use. Default: %(default)s",
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode. Saves conversation to conversation.json after each turn.",
    )

    return parser.parse_args()
