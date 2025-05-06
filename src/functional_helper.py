from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from typing import Iterable, Callable, Iterator, Optional, TypeVar

T = TypeVar("T")  # Any Type
U = TypeVar("U")  # Any Other Type


def for_each(action: Callable[[T], None], collection: Iterable[T]) -> None:
    for item in collection:
        action(item)


def for_each_parallel(action: Callable[[T], None], collection: Iterable[T], max_number_of_workers: Optional[int] = None) -> None:
    if max_number_of_workers is None:
        max_number_of_workers = multiprocessing.cpu_count()

    with ThreadPoolExecutor(max_number_of_workers) as pool:
        list(pool.map(action, collection))


def filter_map(filter_function: Callable[[T], bool], map_function: Callable[[T], U], iterable: Iterable[T]) -> Iterator[U]:
    for item in iterable:
        if filter_function(item):
            yield map_function(item)
