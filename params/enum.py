from collections.abc import Generator
from typing import Any

from parameter import ParameterTypeBase


class EnumParameter(ParameterTypeBase):
    def __init__(self) -> None:
        super().__init__("enum")

    def _customValidate(
        self,
        value: Any,
        opcode: str,
        key: str,
        defaultValue: Any,
        paramRange: tuple[float, float] | None = None,
        paramOptions: dict | None = None,
    ) -> Generator[tuple[bool, Any], None, None]:
        yield from self.expectOptions(value, opcode, key, paramOptions, defaultValue)
