import asyncio
from pathlib import Path
from typing import override

from kosong.tooling import CallableTool2, ToolError, ToolOk, ToolReturnValue
from pydantic import BaseModel, Field

from kimi_cli.tools.function_calling.tools import sleep_tool
from kimi_cli.tools.utils import load_desc


class Params(BaseModel):
    seconds: float = Field(
        description="Number of seconds to sleep before returning.",
        ge=0.0,
    )


class SleepTool(CallableTool2[Params]):
    name: str = "SleepTool"
    description: str = load_desc(Path(__file__).parent / "sleep_tool.md", {})
    params: type[Params] = Params

    @override
    async def __call__(self, params: Params) -> ToolReturnValue:
        try:
            result = await asyncio.to_thread(sleep_tool, params.seconds)
            return ToolOk(output=result)
        except Exception as e:
            return ToolError(
                message=f"Failed to execute sleep request. Error: {e}",
                brief="Sleep failed",
            )
