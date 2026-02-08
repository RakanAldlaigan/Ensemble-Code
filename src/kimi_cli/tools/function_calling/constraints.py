import time
import warnings
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


class Constraints:
    time_threshold: float = 1000

    @staticmethod
    def time_execution(
        enabled: bool = False, threshold_seconds: float | None = None
    ) -> Callable[[Callable[P, R]], Callable[P, R | str]]:
        threshold = (
            Constraints.time_threshold if threshold_seconds is None else threshold_seconds
        )

        def _decorate(base_fn: Callable[P, R]) -> Callable[P, R | str]:
            def enhanced_fn(*args: P.args, **kwargs: P.kwargs) -> R | str:
                if not enabled:
                    return base_fn(*args, **kwargs)

                start_time = time.time()
                result = base_fn(*args, **kwargs)
                end_time = time.time()
                elapsed = end_time - start_time
                if elapsed > threshold:
                    warnings.warn(
                        f"Task exceeded time threshold of {threshold} seconds",
                        stacklevel=2,
                    )
                    return (
                        "Request Violated Time Constraint, Execution Time: "
                        + str(elapsed)
                        + " seconds. Exiting"
                    )
                return result

            return enhanced_fn

        return _decorate
