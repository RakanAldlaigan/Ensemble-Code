import asyncio
from pathlib import Path
from typing import override

from kosong.tooling import CallableTool2, ToolError, ToolOk, ToolReturnValue
from pydantic import BaseModel, Field

from kimi_cli.tools.function_calling.toolkit import Toolkit
from kimi_cli.tools.utils import load_desc


class Params(BaseModel):
    prompt: str = Field(description="The coding request to execute in a Python sandbox.")
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Optional max tokens for the underlying model request.",
    )


class ExecuteCode(CallableTool2[Params]):
    name: str = "ExecuteCode"
    description: str = load_desc(Path(__file__).parent / "execute_code.md", {})
    params: type[Params] = Params

    @override
    async def __call__(self, params: Params) -> ToolReturnValue:
        try:
            result = await asyncio.to_thread(Toolkit.execute_code, params.prompt, params.max_tokens)
        except Exception as e:
            return ToolError(
                message=f"Failed to execute code request. Error: {e}",
                brief="Code execution failed",
            )

        if result == -1:
            return ToolError(
                message="The code generation model returned empty code.",
                brief="No code generated",
            )
        return ToolOk(output=str(result))
