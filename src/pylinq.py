from collections import defaultdict
import functools
import itertools
from typing import Any, Generic, Iterable, Callable, Iterator, Optional, TypeVar, cast, overload
from utilities.entrata_library.functional_helpers import for_each, for_each_parallel


T = TypeVar("T")  # Any Type
U = TypeVar("U")  # Any Other Type
K = TypeVar("K")  # Key Type
V = TypeVar("V")  # Value Type


class PyLinq(Generic[T]):
    _items: list[T]

    def __init__(self, iterable: Optional[Iterable[T]] = None) -> None:
        self._items = list(iterable) if iterable is not None else []

    def __add__(self, other: Iterable[T]) -> "PyLinq[T]":
        return self.concat(other)

    def __bool__(self) -> bool:
        return bool(self._items)

    def __contains__(self, item: T) -> bool:
        return self.contains(item)

    def __delitem__(self, index: int | slice) -> None:
        del self._items[index]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PyLinq):
            return self._items == other._items
        elif isinstance(other, list):
            return self._items == other
        raise NotImplementedError(f"Not able to compare PyLinq to {type(other)}")

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> 'PyLinq[T]': ...

    def __getitem__(self, index: int | slice) -> T | 'PyLinq[T]':
        if isinstance(index, slice):
            return PyLinq(self._items[index])
        elif isinstance(index, int):
            return self._items[index]
        raise TypeError("The index passed in must be an integer or a slice")

    def __iter__(self) -> Iterator:
        return iter(self._items)

    def __iadd__(self, other: Iterable[T]) -> 'PyLinq[T]':
        self._items.extend(other)
        return self

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"PyLinq({repr(self._items)})"

    def __setitem__(self, index: int | slice, value: T | Iterable[T]) -> None:
        if isinstance(index, slice):
            if not isinstance(value, Iterable):
                raise TypeError("Can only assign an iterable to a slice")
            self._items[index] = value  # type: ignore
        else:
            self._items[index] = cast(T, value)

    def __str__(self) -> str:
        return str(self._items)

    def add(self, item: T) -> "PyLinq[T]":
        """Add an item to the list"""
        return self.append(item)

    def aggregate(self, initial: U, func: Callable[[U, T], U]) -> U:
        """Aggregate elements using a function. A wrapper around the 'reduce' function"""
        return self.reduce(initial, func)

    def all(self, predicate: Callable[[T], bool]) -> bool:
        """Whether all elements satisfy the predicate."""
        return all(predicate(item) for item in self._items)

    def any(self, predicate: Optional[Callable[[T], bool]] = None) -> bool:
        """Whether any element satisfies the predicate, or any element exists if no predicate."""
        if predicate is None:
            return len(self._items) > 0
        return any(predicate(item) for item in self._items)

    def append(self, item: T) -> "PyLinq[T]":
        self._items.append(item)
        return PyLinq(self._items)

    def average(self, selector: Optional[Callable[[T], int | float]] = None) -> float:
        """Average of elements, optionally using a selector."""
        if not self._items:
            raise ValueError("Cannot compute average of empty sequence")
        if selector:
            values = [selector(item) for item in self._items]
            return sum(values) / len(values)
        return sum(self._items) / len(self._items)  # type: ignore

    def clear(self) -> None:
        """Clear the list"""
        self._items.clear()

    def concat(self, other: Iterable[T]) -> 'PyLinq[T]':
        """Concatenate with another iterable."""
        return PyLinq(itertools.chain(self._items, other))

    def contains(self, value: T) -> bool:
        """Whether the sequence contains the value."""
        return value in self._items

    def copy(self) -> 'PyLinq[T]':
        """Copy the list"""
        return PyLinq(self._items.copy())

    def count(self) -> int:
        return len(self._items)

    def count_value(self, value: T) -> int:
        """Count occurrences of a specific value in the collection."""
        return self._items.count(value)

    def distinct(self, key_selector: Optional[Callable[[T], Any]] = None) -> 'PyLinq[T]':
        """Return distinct elements, optionally using a key selector."""
        if key_selector is None:
            seen = set()
            result = []
            for item in self._items:
                if item not in seen:
                    seen.add(item)
                    result.append(item)
            return PyLinq(result)
        else:
            seen = set()
            result = []
            for item in self._items:
                key = key_selector(item)
                if key not in seen:
                    seen.add(key)
                    result.append(item)
            return PyLinq(result)

    def element_at(self, index: int) -> T:
        """Element at the specified index."""
        if 0 <= index < len(self._items):
            return self._items[index]
        raise IndexError(f"Index {index} out of range")

    def element_at_or_default(self, index: int, default: T) -> T:
        """Element at the specified index, or default if out of range."""
        if 0 <= index < len(self._items):
            return self._items[index]
        return default

    def except_for(self, other: Iterable[T]) -> 'PyLinq[T]':
        """Return elements from this sequence that aren't in the other sequence."""
        other_set = set(other)
        return PyLinq([item for item in self._items if item not in other_set])

    def extend(self, other: Iterable[T]) -> 'PyLinq[T]':
        return self.concat(other)

    def filter(self, predicate: Callable[[T], bool]) -> "PyLinq[T]":
        return self.where(predicate)

    def first(self, predicate: Optional[Callable[[T], bool]] = None) -> T:
        """First element matching predicate, or first element if no predicate."""
        if len(self._items) == 0:
            raise IndexError("No items list")
        if predicate is None:
            return self._items[0]
        for item in self._items:
            if predicate(item):
                return item
        raise RuntimeError("No value matching predicate")

    def first_or_default(self, default: T, predicate: Optional[Callable[[T], bool]] = None) -> T:
        """First element matching predicate, or default if none found."""
        if not self._items:
            return default
        if predicate is None:
            return self._items[0]
        for item in self._items:
            if predicate(item):
                return item
        return default

    def for_each(self, action: Callable[[T], None]) -> None:
        """Perform an action on each element."""
        for_each(action, self._items)

    def for_each_parallel(self, action: Callable[[T], None], max_workers: Optional[int] = None) -> None:
        for_each_parallel(action, self._items, max_workers)

    def index(self, item: T) -> int:
        """Return the index of the first occurrence of the item."""
        try:
            return self._items.index(item)
        except ValueError:
            return -1

    def insert(self, index: int, item: T) -> 'PyLinq[T]':
        """Insert an item at the specified index."""
        self._items.insert(index, item)
        return self

    def inner_join(self, inner: Iterable[U], outer_key_selector: Callable[[T], K], inner_key_selector: Callable[[U], K], result_selector: Callable[[T, U], V]) -> 'PyLinq[V]':
        """Inner join with another sequence."""
        inner_dict = defaultdict(list)
        for item in inner:
            key = inner_key_selector(item)
            inner_dict[key].append(item)

        result = []
        for outer_item in self._items:
            outer_key = outer_key_selector(outer_item)
            if outer_key in inner_dict:
                for inner_item in inner_dict[outer_key]:
                    result.append(result_selector(outer_item, inner_item))

        return PyLinq(result)

    def intersect(self, other: Iterable[T]) -> 'PyLinq[T]':
        """Return elements that exist in both sequences."""
        other_set = set(other)
        return PyLinq([item for item in self._items if item in other_set])

    def join(self, separator: str) -> str:
        """
        Joins the elements in the sequence using the specified separator.
        Each element is converted to a string before joining.

        Args:
            separator: The string to use as a separator

        Returns:
            A string of all elements joined with the separator
        """
        return separator.join(str(item) for item in self._items)

    def group_by(self, key_selector: Callable[[T], K]) -> 'PyLinq[tuple[K, PyLinq[T]]]':
        """Group elements by key."""
        groups: dict[K, list[T]] = defaultdict(list)
        for item in self._items:
            key = key_selector(item)
            groups[key].append(item)
        return PyLinq([(key, PyLinq(items)) for key, items in groups.items()])

    def last(self, predicate: Optional[Callable[[T], bool]] = None) -> T:
        """Last element matching predicate, or last element if no predicate."""
        if not self._items:
            raise ValueError("Sequence contains no elements")
        if predicate is None:
            return self._items[-1]
        for item in reversed(self._items):
            if predicate(item):
                return item
        raise ValueError("No element satisfies the condition")

    def last_or_default(self, default: T, predicate: Optional[Callable[[T], bool]] = None) -> T:
        """Last element matching predicate, or default if none found."""
        if not self._items:
            return default
        if predicate is None:
            return self._items[-1]
        for item in reversed(self._items):
            if predicate(item):
                return item
        return default

    def map(self, selector: Callable[[T], U]) -> "PyLinq":
        return PyLinq(map(selector, self._items))

    def max(self, selector: Optional[Callable[[T], Any]] = None) -> T:
        """Maximum element, optionally using a selector."""
        if not self._items:
            raise ValueError("Cannot determine maximum of empty sequence")
        if selector:
            return max(self._items, key=selector)
        return max(self._items)  # type: ignore

    def min(self, selector: Optional[Callable[[T], Any]] = None) -> T:
        """Minimum element, optionally using a selector."""
        if not self._items:
            raise ValueError("Cannot determine minimum of empty sequence")
        if selector:
            return min(self._items, key=selector)
        return min(self._items)  # type: ignore

    def missing_from(self, other: Iterable[T]) -> 'PyLinq[T]':
        """Return elements from this sequence that aren't in the other sequence."""
        return self.except_for(other)

    def order_by(self, key_selector: Optional[Callable[[T], Any]] = None) -> "PyLinq[T]":
        """Sorts the objects by the key"""
        return self.sort(key_selector)

    def order_by_descending(self, key_selector: Optional[Callable[[T], Any]] = None) -> 'PyLinq[T]':
        """Sorts the objects by the key in reverse order"""
        return PyLinq(sorted(self._items, key=key_selector, reverse=True))  # type: ignore

    def pop(self) -> T:
        """Remove the last item from the list and return it"""
        return self._items.pop()

    def reduce(self, initial: U, func: Callable[[U, T], U]) -> U:
        return functools.reduce(func, self._items, initial)

    def remove(self, item: T) -> 'PyLinq[T]':
        """Remove the first occurrence of the item"""
        self._items.remove(item)
        return self

    def reverse(self) -> 'PyLinq[T]':
        """Reverse the list"""
        return PyLinq(self._items[::-1])

    def select(self, selector: Callable[[T], U]) -> "PyLinq[U]":
        return self.map(selector)

    def select_many(self, selector: Callable[[T], Iterable[U]]) -> 'PyLinq[U]':
        return PyLinq(itertools.chain.from_iterable(selector(item) for item in self._items))

    def sequence_equal(self, other: Iterable[T]) -> bool:
        """Whether this sequence equals another sequence."""
        other_list = list(other)
        if len(self._items) != len(other_list):
            return False
        return all(a == b for a, b in zip(self._items, other_list))

    def single(self, predicate: Optional[Callable[[T], bool]] = None) -> T:
        """The only element that matches the predicate, or only element if no predicate."""
        if predicate is None:
            if len(self._items) != 1:
                raise ValueError(f"Sequence contains {len(self._items)} elements, not exactly one")
            return self._items[0]
        result = None
        found = False
        for item in self._items:
            if predicate(item):
                if found:
                    raise ValueError("More than one element satisfies the condition")
                result = item
                found = True

        if not found:
            raise ValueError("No element satisfies the condition")
        return result  # type: ignore

    def single_or_default(self, default: T, predicate: Optional[Callable[[T], bool]] = None) -> T:
        """The only element that matches the predicate, or default."""
        try:
            return self.single(predicate)
        except ValueError:
            return default

    def skip(self, count: int) -> 'PyLinq[T]':
        """Skip the specified number of elements."""
        return PyLinq(self._items[count:])

    def skip_while(self, predicate: Callable[[T], bool]) -> 'PyLinq[T]':
        """Skip elements while the predicate is true, then take the rest."""
        for i, item in enumerate(self._items):
            if not predicate(item):
                return PyLinq(self._items[i:])
        return PyLinq([])

    def sort(self, key_selector: Optional[Callable[[T], Any]] = None) -> "PyLinq[T]":
        """Sorts the objects by the key"""
        return PyLinq(sorted(self._items, key=key_selector))  # type: ignore

    def sum(self, selector: Optional[Callable[[T], int | float]] = None) -> int | float:
        """Sum of elements, optionally using a selector."""
        if selector:
            return sum(selector(item) for item in self._items)
        return sum(self._items)  # type: ignore

    def take(self, count: int) -> 'PyLinq[T]':
        """Take `count` number of items and form new object with those items"""
        return PyLinq(self._items[:count])

    def take_while(self, predicate: Callable[[T], bool]) -> 'PyLinq[T]':
        """Take elements while the predicate is true, then stop."""
        result = []
        for item in self._items:
            if predicate(item):
                result.append(item)
            else:
                break
        return PyLinq(result)

    def to_dict(self, key_selector: Callable[[T], K], value_selector: Optional[Callable[[T], V]] = None) -> dict[K, V]:
        """Convert to a Python dictionary."""
        if value_selector:
            return {key_selector(item): value_selector(item) for item in self._items}
        return {key_selector(item): item for item in self._items}  # type: ignore

    def to_list(self) -> list[T]:
        """Convert to a list of items"""
        return list(self._items)

    def to_tuple(self) -> tuple[T, ...]:
        """Convert to a tuple of items"""
        return tuple(self._items)

    def to_set(self) -> set[T]:
        """Convert to a set of items"""
        return set(self._items)

    def union(self, other: Iterable[T]) -> 'PyLinq[T]':
        """Return the union of two sequences."""
        return PyLinq(set(self._items).union(other))

    def where(self, predicate: Callable[[T], bool]) -> "PyLinq[T]":
        return PyLinq([item for item in self._items if predicate(item)])

    def zip(self, other: Iterable[U]) -> 'PyLinq[tuple[T, U]]':
        return PyLinq(zip(self._items, other))
