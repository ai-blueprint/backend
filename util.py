from typing import Iterable, TypeVar


T = TypeVar("T")


def putIntoList(data: Iterable[T] | T) -> list[T]:
    if isinstance(data, Iterable):
        return [*data]
    else:
        return [data]
