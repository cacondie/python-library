from unittest import TestCase, mock
import time
from src.functional_helpers import (
    for_each,
    filter_map,
    for_each_concurrent,
    for_each_expanded,
    for_each_expanded_multiple_actions,
    map_concurrent,
    filter_concurrent,
    select_many,
    select_many_and_map,
    select_many_and_map_sub_elements
)


class FunctionHelpersTests(TestCase):
    def test_for_each(self):
        my_list = [1, 2, 3, 4, 5, 6]
        my_func = mock.MagicMock(name="mock_func")
        for_each(my_func, my_list)

        self.assertEqual(my_func.call_count, 6)

    def test_for_each_example(self):
        def sample_function(number: int) -> None:
            number += 4
            print(number)

        my_list = [1, 2, 3, 4, 5, 6]
        for_each(sample_function, my_list)

    def test_filter_map(self):
        number_list = [1, 2, 3, 4, 5, 6]
        actual = filter_map(lambda n: n % 2 == 0, lambda n: str(n * -1), number_list)

        self.assertListEqual(list(actual), ["-2", "-4", "-6"])

        random_words = ["Any", "Boy", "Girl", "Python", "Data", "entrata"]

        actual2 = filter_map(lambda word: "y" in word, lambda x: x[0], random_words)

        self.assertListEqual(list(actual2), ["A", "B", "P"])

    def test_for_each_concurrent(self):
        my_list = [1, 2, 3, 4, 5, 6]

        def simple_action(_):
            pass
        for_each_concurrent(simple_action, my_list, max_number_of_workers=2)

    def test_for_each_expanded(self):
        data = [
            {"category": "A", "items": [1, 2]},
            {"category": "B", "items": [3]},
            {"category": "C", "items": []},  # Expander returns empty
            {"category": "D"}  # Expander might get None for items
        ]

        def expander_func(record):
            return record.get("items")  # Handles missing "items" key gracefully -> None

        processed_pairs = []

        def action_func(record, item):
            processed_pairs.append((record["category"], item))

        for_each_expanded(expander_func, action_func, data)

        expected = [
            ("A", 1),
            ("A", 2),
            ("B", 3)
        ]
        self.assertListEqual(processed_pairs, expected)

        # Test with empty initial collection
        processed_pairs_empty = []
        for_each_expanded(expander_func, lambda r, i: processed_pairs_empty.append((r, i)), [])
        self.assertListEqual(processed_pairs_empty, [])

    def test_for_each_expanded_multiple_actions(self):
        data = [
            {"id": 1, "values": ["x", "y"]},
            {"id": 2, "values": ["z"]}
        ]

        def expander(d):
            return d["values"]

        action1_results = []

        def action1(record, value):
            action1_results.append((record["id"], value + "_action1"))

        action2_results = []

        def action2(record, value):
            action2_results.append((record["id"], value + "_action2"))

        actions = [action1, action2]
        for_each_expanded_multiple_actions(expander, actions, data)

        expected_action1 = [(1, "x_action1"), (1, "y_action1"), (2, "z_action1")]
        expected_action2 = [(1, "x_action2"), (1, "y_action2"), (2, "z_action2")]

        self.assertListEqual(action1_results, expected_action1)
        self.assertListEqual(action2_results, expected_action2)

        # Test with no actions
        action1_results_no_actions = []

        for_each_expanded_multiple_actions(expander, [], data)  # Empty list of actions
        self.assertListEqual(action1_results_no_actions, [], "Should not run any actions if actions list is empty")

        # Test with empty data
        action1_results_empty_data = []
        actions_for_empty = [lambda r, v: action1_results_empty_data.append((r, v))]
        for_each_expanded_multiple_actions(expander, actions_for_empty, [])
        self.assertListEqual(action1_results_empty_data, [])

    def test_map_concurrent(self):
        data = [1, 2, 3, 4, 5]

        def transform_func(x):
            time.sleep(0.001)  # Simulate a tiny bit of I/O-bound work
            return x * 2

        results = map_concurrent(transform_func, data, max_number_of_workers=2)
        self.assertListEqual(results, [2, 4, 6, 8, 10])

        # Test with empty list
        empty_results = map_concurrent(transform_func, [], max_number_of_workers=2)
        self.assertListEqual(empty_results, [])

        # Test with a generator
        def num_generator(n):
            for i in range(1, n + 1):
                yield i

        generator_results = map_concurrent(transform_func, num_generator(3), max_number_of_workers=2)
        self.assertListEqual(generator_results, [2, 4, 6])

    def test_filter_concurrent(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8]

        def predicate_func(x):
            time.sleep(0.001)  # Simulate a tiny bit of I/O-bound work
            return x % 2 == 0

        results = filter_concurrent(predicate_func, data, max_number_of_workers=2)
        self.assertListEqual(results, [2, 4, 6, 8])

        # Test with empty list
        empty_results = filter_concurrent(predicate_func, [], max_number_of_workers=2)
        self.assertListEqual(empty_results, [])

        # Test with a generator
        def num_generator(n):
            for i in range(1, n + 1):
                yield i

        generator_results = filter_concurrent(predicate_func, num_generator(5), max_number_of_workers=2)
        self.assertListEqual(generator_results, [2, 4])

        # Test where predicate returns False for all
        all_false_results = filter_concurrent(lambda x: False, data, max_number_of_workers=2)
        self.assertListEqual(all_false_results, [])

        # Test where predicate returns True for all
        all_true_results = filter_concurrent(lambda x: True, data, max_number_of_workers=2)
        self.assertListEqual(all_true_results, data)

    def test_select_many_helper(self):
        data = [[1, 2], [3], [], [4, 5]]
        # expander is identity for list of lists
        result = list(select_many(expander=lambda x: x, collection=data)) 
        self.assertListEqual(result, [1, 2, 3, 4, 5])

        class Order:
            def __init__(self, order_id, items):
                self.order_id = order_id
                self.items = items

        orders = [Order(1, ["apple", "banana"]), Order(2, ["cherry"])]
        all_items = list(select_many(expander=lambda o: o.items, collection=orders))
        self.assertListEqual(all_items, ["apple", "banana", "cherry"])

        # Empty input collection
        self.assertListEqual(list(select_many(expander=lambda x: x, collection=[])), [])
        # Expander returns empty iterables for all items
        self.assertListEqual(list(select_many(expander=lambda x: [], collection=[[1], [2]])), [])
        # Collection contains items for which expander returns None (should be handled by itertools.chain.from_iterable if map yields None)
        data_with_nones = [Order(3, None), Order(4,["durian"])]
        items_from_nones = list(select_many(expander=lambda o: o.items if o.items else [], collection=data_with_nones))
        self.assertListEqual(items_from_nones, ["durian"])

    def test_select_many_and_map_helper(self):
        class Department:
            def __init__(self, name, employees):
                self.name = name
                self.employees = employees

        departments = [
            Department("Eng", ["Alice", "Bob"]),
            Department("Sales", ["Charlie"]),
            Department("HR", [])  # Department with no employees
        ]

        expander = lambda dept: dept.employees
        # mapper is the pair_mapper from our previous discussion
        mapper = lambda dept, emp: f"{dept.name}: {emp}" 

        result = list(select_many_and_map(expander=expander, mapper=mapper, collection=departments))
        expected = ["Eng: Alice", "Eng: Bob", "Sales: Charlie"]
        self.assertListEqual(result, expected)

        # Empty input collection
        self.assertListEqual(list(select_many_and_map(expander=expander, mapper=mapper, collection=[])), [])

        # Expander returns empty iterables
        dept_no_emps = [Department("Support", [])]
        self.assertListEqual(list(select_many_and_map(expander=expander, mapper=mapper, collection=dept_no_emps)), [])

    def test_select_many_and_map_sub_elements_helper(self):
        data_nested_lists = [[1, 2], [3, 4, 5], [], [6]]

        # Expander: identity for list of lists
        # sub_element_mapper: square the number
        result = list(select_many_and_map_sub_elements(
            collection=data_nested_lists, 
            expander=lambda x: x, 
            sub_element_mapper=lambda n: n * n
        ))
        self.assertListEqual(result, [1, 4, 9, 16, 25, 36])

        class Category:
            def __init__(self, name, products):
                self.name = name 
                self.products = products 

        categories = [
            Category("Fruits", [{"name": "Apple", "price": 1.0}, {"name": "Banana", "price": 0.5}]),
            Category("Drinks", [{"name": "Cola", "price": 1.5}]),
            Category("Dairy", [])  # Category with no products
        ]

        product_prices = list(select_many_and_map_sub_elements(
            collection=categories,
            expander=lambda cat: cat.products,
            sub_element_mapper=lambda prod: prod["price"]
        ))
        self.assertListEqual(product_prices, [1.0, 0.5, 1.5])

        # Empty input collection
        self.assertListEqual(list(select_many_and_map_sub_elements([], lambda x: x, lambda y: y)), [])
        # Expander returns empty iterables
        self.assertListEqual(list(select_many_and_map_sub_elements([[], []], lambda x: x, lambda y: y)), [])
