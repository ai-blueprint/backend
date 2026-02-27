from collections.abc import Generator
from typing import Any

from parameter import ParameterTypeBase


class FloatParameter(ParameterTypeBase):
    def __init__(self) -> None:
        super().__init__("float", (int, float), bool)

    def _customValidate(
        self,
        value: Any,
        opcode: str,
        key: str,
        defaultValue: Any,
        paramRange: tuple[float, float] | None = None,
        paramOptions: dict | None = None,
    ) -> Generator[tuple[bool, Any], None, None]:
        yield from self.expectType(value, opcode, key, defaultValue)
        yield from self.expectRange(value, opcode, key, paramRange)
