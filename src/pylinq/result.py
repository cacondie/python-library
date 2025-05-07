from typing import Callable, Optional, Generic, TypeVar


T = TypeVar("T")  # Success type
E = TypeVar("E")  # Error type
U = TypeVar("U")  # Other

class Result(Generic[T, E]):
    _value: Optional[T]
    _error: Optional[E]
    is_successful: bool

    def __init__(self, value: Optional[T] = None, error: Optional[E] = None, is_successful: bool = False) -> None:
        if (value is not None) and (error is not None):
            raise ValueError("Result cannot have both a value and an error.")
        self._value = value
        self._error = error
        self.is_successful = is_successful

    def is_ok(self) -> bool:
        """Returns True if the operation was successful and false if it wasn't"""
        return self.is_successful

    def is_err(self) -> bool:
        """Returns True if the operation resulted in an error and false if it didn't"""
        return not self.is_successful

    def map(self, function: Callable[[T], U]) -> 'Result[U, E]':
        """
        Maps a Result<T, E> to Result<U, E> by applying a function to the contained Ok value.
        Leaves Err values unchanged.
        """
        if self.is_ok():
            if self._value is None:
                raise RuntimeError("Value someone got to be None")
            return Result(value=function(self._value), is_successful=True)
        else:
            return Result(error=self._error, is_successful=False)

    def and_then(self, function: Callable[[T], 'Result[U, E]']) -> 'Result[U, E]':
        """
        Returns the result of applying function to the contained Ok value, or returns the Err value.
        Similar to map, but the mapping function returns a Result, this returns the underlying value.
        """
        if self.is_ok():
            if self._value is None:
                raise RuntimeError("Value somehow got to be None")
            return function(self._value)
        else:
            return Result(error=self._error, is_successful=False)

    def unwrap(self) -> T:
        """Returns the value if it's Ok, otherwise raises an error."""
        if self.is_err():
            raise ValueError(f"Called unwrap on an error: {self._error}")
        if self._value is None:
            raise RuntimeError("value somehow got to be None")
        return self._value

    def unwrap_or(self, default: T) -> T:
        """Returns the value if it's Ok, otherwise returns a default value."""
        return self._value if self.is_ok() and self._value is not None else default

    def unwrap_or_else(self, fn: Callable[[E], T]) -> T:
        """
        Returns the contained Ok value if it is successful. If it is an error runs function on error.
        """
        if self.is_ok():
            if self._value is None:
                raise RuntimeError("Value somehow got to be None")
            return self._value
        else:
            if self._error is None:
                raise RuntimeError("Error somehow got to be None")
            return fn(self._error)
        
    def unwrap_err(self) -> E:
        """Returns the error if it's an Err, otherwise raises an error."""
        if self.is_ok():
            raise ValueError("Called unwrap_err on an Ok value.")
        if self._error is None:
            raise RuntimeError("Error somehow got to be None")
        return self._error
    
    def contains(self, value: T) -> bool:
        """
        Returns true if the result is Ok and contains the given value.
        """
        return self.is_ok() and self._value == value

    def contains_err(self, error: E) -> bool:
        """
        Returns true if the result is Err and contains the given error.
        """
        return self.is_err() and type(self._error).__name__ == type(error).__name__

    def or_else(self, function: Callable[[E], 'Result[T, U]']) -> 'Result[T, U]':
        """
        Returns self if it's Ok, otherwise calls function with the error and returns the result.
        """
        if self.is_ok():
            return Result(value=self._value, is_successful=True)
        else:
            if self._error is None:
                raise RuntimeError("Error somehow got to be None")
            return function(self._error)

    def map_err(self, function: Callable[[E], U]) -> 'Result[T, U]':
        """
        Maps a Result<T, E> to Result<T, U> by applying a function to the error value.
        Leaves Ok values unchanged.
        """
        if self.is_ok():
            return Result(value=self._value, is_successful=True)
        else:
            if self._error is None:
                raise RuntimeError("Error someone got to be None")
            return Result(error=function(self._error), is_successful=False)

    def expect(self, message: str) -> T:
        """
        Returns the contained Ok value or raises an exception with the provided message.
        """
        if self.is_err():
            raise ValueError(f"{message}: {self._error}")
        if self._value is None:
            raise RuntimeError("Value somehow got to be None")
        return self._value

    def expect_err(self, msg: str) -> E:
        """
        Returns the contained Err value or raises an exception with the provided message.
        """
        if self.is_ok():
            raise ValueError(f"{msg}: {self._value}")
        if self._error is None:
            raise RuntimeError("Error somehow got to be None")
        return self._error
    
    def ok(self) -> Optional[T]:
        """
        Converts from Result<T, E> to Optional<T>, discarding the error.
        """
        return self._value if self.is_ok() else None

    def err(self) -> Optional[E]:
        """
        Converts from Result<T, E> to Optional<E>, discarding the value.
        """
        return self._error if self.is_err() else None

    def __repr__(self) -> str:
        return f"Result(is_successful: {self.is_successful} -- _value: {self._value} -- _error: {self._error})"


def Ok(value: Optional[T] = None) -> Result[T, E]:
    return Result(value=value, is_successful=True)


def Err(error: E) -> Result[T, E]:
    return Result(error=error, is_successful=False)