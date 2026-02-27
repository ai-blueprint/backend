from collections.abc import Generator
from typing import Any

from parameter import ParameterTypeBase


class StringParameter(ParameterTypeBase):
    def __init__(self) -> None:
        # expect a plain string
        super().__init__("str", str)

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
