import json
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel


@dataclass(slots=True, frozen=True)
class ToolResult:
    """Result of a tool execution."""

    status_code: int
    output: str
    error_message: str | None = None

    def __post_init__(self):
        if self.status_code != 0 and not self.error_message:
            raise ValueError(
                "error_message must be provided if status_code is non-zero"
            )


def pydantic_to_tool_params(
    name: str,
    description: str,
    parameters_model: type[BaseModel],
) -> dict:
    """Convert a Pydantic model to tool parameters dict."""

    is_pydantic = isinstance(parameters_model, type) and issubclass(
        parameters_model, BaseModel
    )
    if not is_pydantic:
        raise ValueError("parameters_model must be a Pydantic model class")

    model_schema = parameters_model.model_json_schema()
    if "title" in model_schema:
        # Remove title to avoid issues with some LLM parsers
        del model_schema["title"]

    # For strict mode, all properties must be in required array
    # and additionalProperties must be false
    if "properties" in model_schema:
        model_schema["required"] = list(model_schema["properties"].keys())
    model_schema["additionalProperties"] = False

    return {
        "name": name,
        "description": description,
        "parameters": model_schema,
        "strict": True,
    }


class ListFilesToolParameters(BaseModel, extra="forbid"):
    folder: str


ListFilesTool = pydantic_to_tool_params(
    name="list_files",
    description=(
        "List top-level files in a folder (non-recursive). "
        "Input should be a valid folder path."
    ),
    parameters_model=ListFilesToolParameters,
)


def list_files(folder: Path) -> ToolResult:
    """List top-level files in a folder (non-recursive)."""
    try:
        return ToolResult(
            output=json.dumps([p for p in folder.iterdir()], default=str),
            status_code=0,
        )
    except Exception as e:
        return ToolResult(status_code=1, output="", error_message=str(e))


class ReadFileToolParameters(BaseModel, extra="forbid"):
    file_path: str
    line_start: int | None = None
    line_end: int | None = None


ReadFileTool = pydantic_to_tool_params(
    name="read_file",
    description="Read the content of a file. Input should be a valid file path. Optional line_start and line_end can be provided to read specific lines.",
    parameters_model=ReadFileToolParameters,
)


def read_file(
    file_path: Path, line_start: int | None = None, line_end: int | None = None
) -> ToolResult:
    """Read the content of a file."""
    if not file_path.exists():
        return ToolResult(
            status_code=1,
            output="",
            error_message=f"File {file_path} does not exist.",
        )

    if not file_path.is_file():
        return ToolResult(
            status_code=1,
            output="",
            error_message=f"Path {file_path} is not a file.",
        )

    if line_start is not None and line_start < 1:
        return ToolResult(
            status_code=1,
            output="",
            error_message="line_start must be >= 1.",
        )

    if line_end is not None and line_end < 1:
        return ToolResult(
            status_code=1,
            output="",
            error_message="line_end must be >= 1.",
        )

    if (line_start is not None) and (line_end is None):
        return ToolResult(
            status_code=1,
            output="",
            error_message="line_end must be provided if line_start is provided.",
        )
    if (line_end is not None) and (line_start is None):
        return ToolResult(
            status_code=1,
            output="",
            error_message="line_start must be provided if line_end is provided.",
        )

    if (line_start is not None and line_end is not None) and (line_end < line_start):
        return ToolResult(
            status_code=1,
            output="",
            error_message="line_end must be greater than or equal to line_start.",
        )

    try:
        content = file_path.read_text()
        if line_start is not None or line_end is not None:
            lines = content.splitlines()
            line_start_idx = line_start - 1 if line_start and line_start > 0 else 0
            line_end_idx = (
                line_end if line_end and line_end <= len(lines) else len(lines)
            )
            content = "\n".join(lines[line_start_idx:line_end_idx])
        return ToolResult(status_code=0, output=content)
    except Exception as e:
        return ToolResult(status_code=1, output="", error_message=str(e))


class GrepPatternToolParameters(BaseModel, extra="forbid"):
    base_path: str
    pattern: str


GrepPatternTool = pydantic_to_tool_params(
    name="grep_pattern",
    description="Search for a regex pattern in files under a base path. Input should be a valid base path and a regex pattern.",
    parameters_model=GrepPatternToolParameters,
)


def grep_pattern(base_path: Path, pattern: str) -> ToolResult:
    """Search for a regex pattern in files under base_path."""
    compiled_pattern = re.compile(pattern)
    matches = []

    try:
        for file_path in base_path.rglob("*"):
            if file_path.is_file():
                try:
                    with file_path.open("r") as f:
                        for line_number, line in enumerate(f, start=1):
                            if compiled_pattern.search(line):
                                matches.append(
                                    f"{file_path}:{line_number}:{line.strip()}"
                                )
                except Exception:
                    # Skip files that can't be read
                    continue

        return ToolResult(
            status_code=0,
            output=json.dumps(matches, indent=2),
        )
    except Exception as e:
        return ToolResult(status_code=1, output="", error_message=str(e))
