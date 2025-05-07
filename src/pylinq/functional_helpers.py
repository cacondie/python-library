from concurrent.futures import ThreadPoolExecutor
import itertools
import multiprocessing
from typing import Iterable, Callable, Iterator, Optional, TypeVar

T = TypeVar("T")  # Any Type
U = TypeVar("U")  # Any Other Type
V = TypeVar("V")  # Any Other Type


def for_each(action: Callable[[T], None], collection: Iterable[T]) -> None:
    for item in collection:
        action(item)


def for_each_expanded(expander: Callable[[T], Iterable[U]], action: Callable[[T, U], None], collection: Iterable[T]) -> None:
    """
    For each element in the collection, expands it into a sequence of sub-elements
    and performs an action on each (original_element, sub_element) pair.

    Args:
        expander: A function that takes an element of type T and returns an
                    iterable of elements of type U.
        action: A function that takes an element of type T (original_element)
                and an element of type U (sub_element) and performs an action.
    """
    for item_t in collection:
        expanded_items_u = expander(item_t)
        if expanded_items_u:
            for item_u in expanded_items_u:
                action(item_t, item_u)


def for_each_expanded_multiple_actions(expander: Callable[[T], Iterable[U]], actions: Iterable[Callable[[T, U], None]], collection: Iterable[T]) -> None:
    """
    For each element in the collection, expands it into a sequence of sub-elements
    and performs an action on each (original_element, sub_element) pair.

    Args:
        expander: A function that takes an element of type T and returns an
                    iterable of elements of type U.
        actions: An Iterable of functions that takes an element of type T (original_element)
                and an element of type U (sub_element) and performs an action.
        collection: The collection of elements to process.
    """
    for item_t in collection:
        expanded_items_u = expander(item_t)
        if expanded_items_u:
            for item_u in expanded_items_u:
                for action in actions:
                    action(item_t, item_u)


def for_each_concurrent(action: Callable[[T], None], collection: Iterable[T], max_number_of_workers: Optional[int] = None) -> None:
    """
    Applies an action to each item in a collection concurrently.
    Best actions that have some sort of waiting (like IO or network calls).
    """
    if max_number_of_workers is None:
        max_number_of_workers = multiprocessing.cpu_count()

    with ThreadPoolExecutor(max_number_of_workers) as pool:
        list(pool.map(action, collection))


def filter_map(filter_function: Callable[[T], bool], map_function: Callable[[T], U], iterable: Iterable[T]) -> Iterator[U]:
    for item in iterable:
        if filter_function(item):
            yield map_function(item)


def map_concurrent(map_function: Callable[[T], U], collection: Iterable[T], max_number_of_workers: Optional[int] = None) -> list[U]:
    """
    Applies a transformation function to each item in a collection concurrently.
    Best functions that have some sort of waiting (like IO or network calls).
    Returns a list of the transformed items in the original order.
    """
    if max_number_of_workers is None:
        max_number_of_workers = multiprocessing.cpu_count()

    with ThreadPoolExecutor(max_workers=max_number_of_workers) as pool:
        return list(pool.map(map_function, collection))


def filter_concurrent(predicate: Callable[[T], bool], collection: Iterable[T], max_number_of_workers: Optional[int] = None) -> list[T]:
    """
    Filters a collection based on a predicate function applied concurrently.
    Best functions that have some sort of waiting (like IO or network calls).
    Returns a list of items for which the predicate returned True, in original order.
    """
    if max_number_of_workers is None:
        max_number_of_workers = multiprocessing.cpu_count()

    collection_list = list(collection)
    if not collection_list:
        return []

    with ThreadPoolExecutor(max_workers=max_number_of_workers) as pool:
        boolean_results = list(pool.map(predicate, collection_list))

    return [item for item, keep in zip(collection_list, boolean_results) if keep]


def select_many(expander: Callable[[T], Iterable[U]], collection: Iterable[T]) -> Iterator[U]:
    """
    Applies an expander function to each element of a collection,
    where the expander returns an iterable, and then flattens (chains) the
    resulting iterables into a single iterator.
    Equivalent to LINQ's SelectMany(expander) or flatMap in other languages.

    Args:
        expander: A function that takes an element of type T and returns an
                  iterable of elements of type U.
        collection: The input collection of elements of type T.

    Returns:
        An iterator yielding elements of type U.
    """
    return itertools.chain.from_iterable(map(expander, collection))


def select_many_and_map(expander: Callable[[T], Iterable[U]], mapper: Callable[[T, U], V], collection: Iterable[T]) -> Iterator[V]:
    for item_t, item_u in _expand_item_pairs(collection, expander):
        yield mapper(item_t, item_u)


def select_many_and_map_sub_elements(collection: Iterable[T], expander: Callable[[T], Iterable[U]], sub_element_mapper: Callable[[U], V]) -> Iterator[V]:
    for _, item_u in _expand_item_pairs(collection, expander):
        yield sub_element_mapper(item_u)


def _expand_item_pairs(collection: Iterable[T], expander: Callable[[T], Iterable[U]]) -> Iterator[tuple[T, U]]:
    for item_t in collection:
        expanded_items_u = expander(item_t)
        if expanded_items_u:
            for item_u in expanded_items_u:
                yield (item_t, item_u)
