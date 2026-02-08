from __future__ import annotations

import asyncio
import json
import time
from typing import Any, cast

from openai import AsyncOpenAI

from . import tools


async def run_agent(prompt: str) -> str:
    client = AsyncOpenAI()
    input_list: list[Any] = [{"role": "user", "content": prompt}]

    while True:
        response = await client.responses.create(
            model="gpt-5",
            tools=cast(Any, tools.TOOLS),
            input=cast(Any, input_list),
        )

        input_list += cast(list[Any], response.output)

        tool_calls: list[Any] = [
            i for i in response.output if getattr(i, "type", None) == "function_call"
        ]
        if not tool_calls:
            return str(response.output_text or "")

        async def do_one(call: Any) -> dict[str, Any]:
            result = await asyncio.to_thread(tools.run_tool, call.name, call.arguments)
            return {
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps({"result": result}),
            }

        outputs = await asyncio.gather(*(do_one(c) for c in tool_calls))
        input_list.extend(outputs)


async def main() -> None:
    prompt = (
        "Call sleep_total three times in parallel: "
        "sleep_tool(seconds=2), sleep_tool(seconds=2), sleep_tool(seconds=2). "
        "Then tell me how long it took total."
    )

    t0 = time.perf_counter()
    final = await run_agent(prompt)
    total = time.perf_counter() - t0

    print("FINAL OUTPUT:\n", final)
    print(f"\nTOTAL WALL TIME: {total:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
