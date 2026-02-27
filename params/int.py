from collections.abc import Generator
from typing import Any

from parameter import ParameterTypeBase, ValidateResult


class IntParameter(ParameterTypeBase):
    def __init__(self) -> None:
        super().__init__("int", (int, float), bool)

    def _customValidate(
        self,
        value: Any,
        opcode: str,
        key: str,
        defaultValue: Any,
        paramRange: tuple[float, float] | None = None,
        paramOptions: dict | None = None,
    ) -> Generator[ValidateResult]:
        yield from self.expectType(value, opcode, key, defaultValue)
        yield from self.expectRange(value, opcode, key, paramRange)
