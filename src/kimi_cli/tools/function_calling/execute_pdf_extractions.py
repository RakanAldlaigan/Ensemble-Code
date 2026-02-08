import asyncio
from pathlib import Path
from typing import override

from kosong.tooling import CallableTool2, ToolError, ToolOk, ToolReturnValue
from pydantic import BaseModel, Field

from kimi_cli.tools.function_calling.toolkit import Toolkit
from kimi_cli.tools.utils import load_desc


class Params(BaseModel):
    prompt: str = Field(description="The extraction request for PDF-focused analysis.")
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Optional max tokens for the underlying model request.",
    )


class ExecutePDFExtractions(CallableTool2[Params]):
    name: str = "ExecutePDFExtractions"
    description: str = load_desc(Path(__file__).parent / "execute_pdf_extractions.md", {})
    params: type[Params] = Params

    @override
    async def __call__(self, params: Params) -> ToolReturnValue:
        try:
            result = await asyncio.to_thread(
                Toolkit.execute_pdf_extractions,
                params.prompt,
                params.max_tokens,
            )
            return ToolOk(output=str(result))
        except Exception as e:
            return ToolError(
                message=f"Failed to execute PDF extraction request. Error: {e}",
                brief="PDF extraction failed",
            )
