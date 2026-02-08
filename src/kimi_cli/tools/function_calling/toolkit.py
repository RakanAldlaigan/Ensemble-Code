from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from openai import OpenAI

from .constraints import Constraints


class Toolkit:
    max_tokens: int | None = None
    time_constraint: bool = False

    def __init__(self, max_tokens: int | None = None, time_constraint: bool = False) -> None:
        Toolkit.max_tokens = max_tokens
        Toolkit.time_constraint = time_constraint

    @staticmethod
    def _effective_max_tokens(max_tokens: int | None, default_value: int | None) -> int | None:
        if max_tokens is not None:
            return max_tokens
        if Toolkit.max_tokens is not None:
            return Toolkit.max_tokens
        return default_value

    @staticmethod
    def _run_with_time_constraint(base_fn: Callable[[], Any]) -> Any:
        wrapped = Constraints.time_execution(enabled=Toolkit.time_constraint)(base_fn)
        return wrapped()

    @staticmethod
    def execute_code(prompt: str, max_tokens: int | None = None) -> str | int:
        def _run() -> str | int:
            client = OpenAI()
            try:
                from e2b_code_interpreter import Sandbox  # type: ignore[reportMissingImports]
            except ModuleNotFoundError as e:
                raise RuntimeError(
                    "execute_code requires e2b_code_interpreter. "
                    "Install it to enable code execution."
                ) from e
            sandbox_cls = cast(Any, Sandbox)
            sbx = sandbox_cls.create()

            system = (
                "You are a helpful assistant that can execute python code in a Jupyter notebook. "
                "Only respond with the code to be executed and nothing else. "
                "Strip backticks in code blocks."
            )

            request: dict[str, Any] = {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            }
            effective_max_tokens = Toolkit._effective_max_tokens(max_tokens, None)
            if effective_max_tokens is not None:
                request["max_tokens"] = effective_max_tokens

            response = cast(Any, client.chat.completions.create(**request))
            code = cast(str | None, response.choices[0].message.content)

            if code:
                execution = sbx.run_code(code)
                result = cast(str, execution.text)
                return result

            return -1

        return Toolkit._run_with_time_constraint(_run)

    @staticmethod
    def execute_calculator(prompt: str, max_tokens: int | None = 128) -> str:
        def _run() -> str:
            client = OpenAI()
            system = """
                    You are a calculator.
                    Return only the final numerical value when this is a calculation request.
                    If it is not a calculation request, return '#'.
                    Do not include explanation.
                    """

            request: dict[str, Any] = {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            }
            effective_max_tokens = Toolkit._effective_max_tokens(max_tokens, 128)
            if effective_max_tokens is not None:
                request["max_tokens"] = effective_max_tokens

            response = cast(Any, client.chat.completions.create(**request))
            result = cast(str | None, response.choices[0].message.content)
            if result:
                return result.strip()
            return "#"

        return Toolkit._run_with_time_constraint(_run)

    @staticmethod
    def execute_pdf_extractions(prompt: str, max_tokens: int | None = None) -> str:
        def _run() -> str:
            client = OpenAI()
            system = "You are a helpful assistant that can extract information from a PDF document."

            request: dict[str, Any] = {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            }
            effective_max_tokens = Toolkit._effective_max_tokens(max_tokens, None)
            if effective_max_tokens is not None:
                request["max_tokens"] = effective_max_tokens

            response = cast(Any, client.chat.completions.create(**request))
            result = cast(str | None, response.choices[0].message.content)
            return result.strip() if result else ""

        return Toolkit._run_with_time_constraint(_run)
