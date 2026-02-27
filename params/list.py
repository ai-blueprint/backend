from collections.abc import Generator
from typing import Any

from parameter import ParameterTypeBase
from params.float import FloatParameter


class ListParameter(ParameterTypeBase):
    def __init__(self) -> None:
        super().__init__("list", list)

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
        cleaned: list[Any] = []
        floatParam = FloatParameter()
        for i, item in enumerate(value):
            for state, dataOrMessage in floatParam.expectType(
                item,
                opcode,
                f"{key}[{i}]",
                defaultValue,
            ):
                if state:
                    cleaned.append(dataOrMessage)
                else:
                    yield False, dataOrMessage
                    cleaned.append(float(item))
                break
        yield True, cleaned
