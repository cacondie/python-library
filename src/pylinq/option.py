from abc import ABC
from typing import Any, Callable, Generic, Literal, TypeVar

T = TypeVar("T")  # Success type
E = TypeVar("E")  # Error type
U = TypeVar("U")  # Other


class Option(Generic[T], ABC):
    value: T

    def is_some(self) -> bool:
        return isinstance(self, Some)

    def is_none(self) -> bool:
        return isinstance(self, None_)

    def unwrap(self) -> T:
        """Returns the value if Some, else raises an exception."""
        if self.is_some():
            return self.value
        raise ValueError("Called unwrap() on None")

    def unwrap_or(self, default: T) -> T:
        """Returns the value if Some, otherwise returns a default."""
        return self.value if isinstance(self, Some) else default

    def unwrap_or_else(self, func: Callable[[], T]) -> T:
        """Returns the value if Some, otherwise calls `func` to get a default."""
        return self.value if isinstance(self, Some) else func()

    def map(self, func: Callable[[T], U]) -> "Option[U]":
        """Applies a function to the value if Some, otherwise returns NoneOption."""
        return Some(func(self.value)) if isinstance(self, Some) else NoneOption

    def and_then(self, func: Callable[[T], "Option[U]"]) -> U:
        """Applies a function that returns an Option if Some, otherwise returns None."""
        return func(self.value) if isinstance(self, Some) else NoneOption


class Some(Option[T]):
    def __init__(self, value: T) -> None:
        self.value = value

    def __repr__(self):
        return f"Some({self.value})"

    def __eq__(self, other_value: Any) -> bool:
        if not isinstance(other_value, Some):
            return False
        if self.value == other_value.unwrap():
            return True
        return False
            

class None_(Option[T]):
    def __repr__(self) -> Literal['NoneOption']:
        return "NoneOption"

    def __str__(self) -> Literal['NoneOption']:
        return self.__repr__()

    def __eq__(self, value) -> bool:
        if value is None:
            return True
        elif isinstance(value, None_):
            return True
        else:
            return False


NoneOption: Option[Any] = None_()