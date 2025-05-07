from unittest import TestCase
from unittest import mock
import threading
import time
from src.pylinq import PyLinq


class PyLinqTests(TestCase):
    def test_basics(self):
        sut = PyLinq(list(range(10))) \
                            .add(10) \
                            .map(lambda i: i-1)
        self.assertEqual(sut.to_list(), [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(sut.sum(), 44)
        self.assertListEqual(sut.filter(lambda i: i % 2 == 0).to_list(), [0, 2, 4, 6, 8])
        self.assertListEqual(sut.where(lambda i: i % 2 == 0).to_list(), [0, 2, 4, 6, 8])
        self.assertListEqual(sut.select(str).to_list(), ['-1', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
        self.assertTupleEqual(sut.where(lambda i: i % 2 == 0).to_tuple(), (0, 2, 4, 6, 8))

    def test_select_many_examples(self):
        # Example 1: Basic flattening of a list of lists
        nested_lists = [
            [1, 2, 3],
            [4, 5],
            [6, 7, 8, 9]
        ]

        result = PyLinq(nested_lists).select_many(lambda x: x).to_list()
        self.assertListEqual(result, [1, 2, 3, 4, 5, 6, 7, 8, 9])

        # Example 2: Working with objects that contain collections
        class Person:
            def __init__(self, name: str, phone_numbers: list[str]):
                self.name = name
                self.phone_numbers = phone_numbers

        people = [
            Person("Alice", ["555-1234", "555-5678"]),
            Person("Bob", ["555-8765"]),
            Person("Charlie", ["555-4321", "555-8901", "555-2345"])
        ]

        # Extract all phone numbers from all people
        all_phone_numbers = PyLinq(people).select_many(lambda p: p.phone_numbers).to_list()
        self.assertListEqual(all_phone_numbers, ['555-1234', '555-5678', '555-8765', '555-4321', '555-8901', '555-2345'])

        # With additional transformation
        formatted_numbers = (PyLinq(people)
                                .select_many(lambda p: [f"{p.name}: {num}" for num in p.phone_numbers])
                                .to_list())
        self.assertListEqual(formatted_numbers, ['Alice: 555-1234', 'Alice: 555-5678', 'Bob: 555-8765', 'Charlie: 555-4321', 'Charlie: 555-8901', 'Charlie: 555-2345'])

        # Example 3: Real-world scenario - Processing nested data structures
        departments = [
            {
                "name": "Engineering",
                "employees": [
                    {"name": "Alice", "skills": ["Python", "SQL", "AWS"]},
                    {"name": "Bob", "skills": ["Java", "C++", "Docker"]}
                ]
            },
            {
                "name": "Marketing",
                "employees": [
                    {"name": "Charlie", "skills": ["SEO", "Content Writing"]},
                    {"name": "David", "skills": ["Social Media", "Analytics", "Design"]}
                ]
            }
        ]
        expected = [
            {'name': 'Alice', 'skills': ['Python', 'SQL', 'AWS']},
            {'name': 'Bob', 'skills': ['Java', 'C++', 'Docker']},
            {'name': 'Charlie', 'skills': ['SEO', 'Content Writing']},
            {'name': 'David', 'skills': ['Social Media', 'Analytics', 'Design']}
        ]
        all_employees = PyLinq(departments).select_many(lambda d: d["employees"]).to_list()
        self.assertListEqual(all_employees, expected)
        # Get all unique skills across all employees in all departments
        all_skills = (
            PyLinq(departments)
                .select_many(lambda d: d["employees"])                 # Flatten departments to employees
                .select_many(lambda e: e["skills"])                    # Flatten employees to skills
                .distinct()                                            # Get unique skills
                .order_by()                                            # Sort alphabetically
                .to_list()
        )
        self.assertListEqual(all_skills, ['AWS', 'Analytics', 'C++', 'Content Writing', 'Design', 'Docker', 'Java', 'Python', 'SEO', 'SQL', 'Social Media'])

    def test_list_builtins_clear_index_copy_insert(self):
        nums = PyLinq([1, 2, 3, 4, 5])
        nums.clear()
        self.assertEqual(nums.to_list(), [])

        nums = PyLinq([1, 2, 3, 4, 5])
        self.assertEqual(nums.index(3), 2)
        self.assertEqual(nums.index(6), -1)

        new_nums = nums.insert(2, 10)
        self.assertEqual(nums.to_list(), [1, 2, 10, 3, 4, 5])
        self.assertEqual(new_nums.to_list(), [1, 2, 10, 3, 4, 5])

        nums = PyLinq([1, 2, 3, 4, 5])
        new_pylinq = nums.copy()
        self.assertEqual(new_pylinq.to_list(), [1, 2, 3, 4, 5])

        nums = PyLinq([1, 2, 3, 4, 5])
        self.assertEqual(nums.pop(), 5)
        self.assertEqual(nums.to_list(), [1, 2, 3, 4])

        nums = PyLinq([1, 2, 3, 4, 5])
        self.assertEqual(nums.remove(3).to_list(), [1, 2, 4, 5])
        
    def test_reverse(self):
        # Test reverse
        nums = PyLinq([1, 2, 3, 4, 5])
        reversed_nums = nums.reverse()
        
        # Check that a new reversed collection is returned
        self.assertEqual(reversed_nums.to_list(), [5, 4, 3, 2, 1])
        
        # Check that the original is unchanged
        self.assertEqual(nums.to_list(), [1, 2, 3, 4, 5])
        
        # Test with complex objects
        people = PyLinq([
            {"name": "Alice", "age": 25},
            {"name": "Bob", "age": 30},
            {"name": "Charlie", "age": 35}
        ])
        
        reversed_people = people.reverse()
        self.assertEqual(reversed_people.to_list()[0]["name"], "Charlie")
        self.assertEqual(reversed_people.to_list()[1]["name"], "Bob")
        self.assertEqual(reversed_people.to_list()[2]["name"], "Alice")

    def test_init_and_iteration(self):
        # Test empty initialization
        empty = PyLinq()
        self.assertEqual(len(empty), 0)
        
        # Test initialization with iterable
        nums = PyLinq([1, 2, 3, 4, 5])
        self.assertEqual(len(nums), 5)
        
        # Test iteration
        items = []
        for item in nums:
            items.append(item)
        self.assertEqual(items, [1, 2, 3, 4, 5])

    def test_operators(self):
        # Test __add__
        nums1 = PyLinq([1, 2, 3])
        nums2 = [4, 5, 6]
        combined = nums1 + nums2
        self.assertEqual(combined.to_list(), [1, 2, 3, 4, 5, 6])
        
        # Test __bool__
        self.assertTrue(bool(PyLinq([1])))
        self.assertFalse(bool(PyLinq()))
        
        # Test __contains__
        nums = PyLinq([1, 2, 3, 4, 5])
        self.assertTrue(3 in nums)
        self.assertFalse(6 in nums)
        
        # Test __delitem__
        nums = PyLinq([1, 2, 3, 4, 5])
        del nums[2]
        self.assertEqual(nums.to_list(), [1, 2, 4, 5])
        del nums[1:3]
        self.assertEqual(nums.to_list(), [1, 5])
        
        # Test __eq__
        self.assertTrue(PyLinq([1, 2, 3]) == PyLinq([1, 2, 3]))
        self.assertTrue(PyLinq([1, 2, 3]) == [1, 2, 3])
        self.assertFalse(PyLinq([1, 2, 3]) == PyLinq([1, 2, 4]))
        with self.assertRaises(NotImplementedError):
            PyLinq([1, 2, 3]) == "not comparable"
        
        # Test __getitem__
        nums = PyLinq([1, 2, 3, 4, 5])
        self.assertEqual(nums[2], 3)
        self.assertEqual(nums[1:4].to_list(), [2, 3, 4])
        with self.assertRaises(TypeError):
            nums["invalid"]
        
        # Test __iadd__
        nums = PyLinq([1, 2, 3])
        nums += [4, 5, 6]
        self.assertEqual(nums.to_list(), [1, 2, 3, 4, 5, 6])
        
        # Test __len__
        self.assertEqual(len(PyLinq([1, 2, 3, 4, 5])), 5)
        
        # Test __repr__ and __str__
        nums = PyLinq([1, 2, 3])
        self.assertEqual(repr(nums), "PyLinq([1, 2, 3])")
        self.assertEqual(str(nums), "[1, 2, 3]")
        
        # Test __setitem__
        nums = PyLinq([1, 2, 3, 4, 5])
        nums[2] = 10
        self.assertEqual(nums.to_list(), [1, 2, 10, 4, 5])
        nums[1:4] = [20, 30, 40]
        self.assertEqual(nums.to_list(), [1, 20, 30, 40, 5])
        with self.assertRaises(TypeError):
            nums[1:3] = 100  # Not iterable

    def test_add_and_append(self):
        # Test add
        nums = PyLinq([1, 2, 3])
        result = nums.add(4)
        self.assertEqual(nums.to_list(), [1, 2, 3, 4])  # Original modified
        self.assertEqual(result.to_list(), [1, 2, 3, 4])  # Returns modified instance
        
        # Test append (alias for add)
        nums = PyLinq([1, 2, 3])
        result = nums.append(4)
        self.assertEqual(nums.to_list(), [1, 2, 3, 4])
        self.assertEqual(result.to_list(), [1, 2, 3, 4])

    def test_aggregate_and_reduce(self):
        nums = PyLinq([1, 2, 3, 4, 5])
        
        # Test aggregate
        sum_result = nums.aggregate(0, lambda acc, x: acc + x)
        self.assertEqual(sum_result, 15)
        
        product_result = nums.aggregate(1, lambda acc, x: acc * x)
        self.assertEqual(product_result, 120)
        
        # Test reduce (alias for aggregate)
        list_of_lists = PyLinq([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        actual_reduce = list_of_lists.reduce([], lambda acc, x: acc + x)
        self.assertEqual(actual_reduce, [1, 2, 3, 4, 5, 6, 7, 8, 9])

    def test_all_and_any(self):
        nums = PyLinq([1, 2, 3, 4, 5])
        
        # Test all
        self.assertTrue(nums.all(lambda x: x > 0))
        self.assertFalse(nums.all(lambda x: x > 3))
        
        # Test any
        self.assertTrue(nums.any(lambda x: x > 3))
        self.assertFalse(nums.any(lambda x: x > 10))
        
        # Test any with no predicate
        self.assertTrue(nums.any())
        self.assertFalse(PyLinq().any())

    def test_average(self):
        nums = PyLinq([1, 2, 3, 4, 5])
        
        # Test average with no selector
        self.assertEqual(nums.average(), 3.0)
        
        # Test average with selector
        people = PyLinq([
            {"name": "Alice", "age": 25},
            {"name": "Bob", "age": 30},
            {"name": "Charlie", "age": 35}
        ])
        self.assertEqual(people.average(lambda p: p["age"]), 30.0)
        
        # Test empty sequence
        with self.assertRaises(ValueError):
            PyLinq().average()

    def test_concat_and_extend(self):
        nums1 = PyLinq([1, 2, 3])
        nums2 = [4, 5, 6]
        
        # Test concat
        result = nums1.concat(nums2)
        self.assertEqual(result.to_list(), [1, 2, 3, 4, 5, 6])
        self.assertEqual(nums1.to_list(), [1, 2, 3])  # Original unchanged
        
        # Test extend (alias for concat)
        result = nums1.extend(nums2)
        self.assertEqual(result.to_list(), [1, 2, 3, 4, 5, 6])
        self.assertEqual(nums1.to_list(), [1, 2, 3])  # Original unchanged

    def test_contains_and_count(self):
        nums = PyLinq([1, 2, 3, 4, 5])
        
        # Test contains
        self.assertTrue(nums.contains(3))
        self.assertFalse(nums.contains(10))
        
        # Test count
        self.assertEqual(nums.count(), 5)
        self.assertEqual(PyLinq().count(), 0)

    def test_count_value(self):
        # Test count_value with numbers
        nums = PyLinq([1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5])
        self.assertEqual(nums.count_value(1), 1)
        self.assertEqual(nums.count_value(2), 2)
        self.assertEqual(nums.count_value(3), 3)
        self.assertEqual(nums.count_value(4), 4)
        self.assertEqual(nums.count_value(5), 5)
        self.assertEqual(nums.count_value(6), 0)  # Not in list
        
        # Test count_value with strings
        words = PyLinq(["apple", "banana", "apple", "cherry", "apple", "banana"])
        self.assertEqual(words.count_value("apple"), 3)
        self.assertEqual(words.count_value("banana"), 2)
        self.assertEqual(words.count_value("cherry"), 1)
        self.assertEqual(words.count_value("date"), 0)  # Not in list
        
        # Test with empty list
        empty = PyLinq([])
        self.assertEqual(empty.count_value(42), 0)

    def test_distinct(self):
        # Test distinct with no key selector
        nums = PyLinq([1, 2, 2, 3, 3, 3, 4, 5, 5])
        self.assertEqual(nums.distinct().to_list(), [1, 2, 3, 4, 5])
        
        # Test distinct with key selector
        people = PyLinq([
            {"name": "Alice", "dept": "IT"},
            {"name": "Bob", "dept": "HR"},
            {"name": "Charlie", "dept": "IT"},
            {"name": "David", "dept": "Finance"}
        ])
        
        # Get distinct departments
        distinct_depts = people.map(lambda p: p["dept"]) \
                               .distinct() \
                               .to_list()
        self.assertListEqual(distinct_depts, ["IT", "HR", "Finance"])
        one_per_dept = people.distinct(lambda p: p["dept"]).to_list()
        self.assertEqual(len(one_per_dept), 3)
        self.assertTrue(all(d in one_per_dept for d in [
            {"name": "Alice", "dept": "IT"},
            {"name": "Bob", "dept": "HR"},
            {"name": "David", "dept": "Finance"}
        ]))

    def test_element_at(self):
        nums = PyLinq([1, 2, 3, 4, 5])
        
        # Test element_at
        self.assertEqual(nums.element_at(2), 3)
        with self.assertRaises(IndexError):
            nums.element_at(10)
        
        # Test element_at_or_default
        self.assertEqual(nums.element_at_or_default(2, 0), 3)
        self.assertEqual(nums.element_at_or_default(100, 42), 42)

    def test_except_for(self):
        nums1 = PyLinq([1, 2, 3, 4, 5])
        nums2 = [3, 4, 5, 6, 7]
        
        # Test except_for
        result = nums1.except_for(nums2)
        result2 = nums1.missing_from(nums2)
        self.assertEqual(result.to_list(), [1, 2])
        self.assertEqual(result2.to_list(), [1, 2])

    def test_filter_and_where(self):
        nums = PyLinq([1, 2, 3, 4, 5])
        
        # Test filter
        self.assertEqual(nums.filter(lambda x: x % 2 == 0).to_list(), [2, 4])
        
        # Test where (alias for filter)
        self.assertEqual(nums.where(lambda x: x % 2 == 0).to_list(), [2, 4])

    def test_first_and_first_or_default(self):
        nums = PyLinq([1, 2, 3, 4, 5])
        
        # Test first with no predicate
        self.assertEqual(nums.first(), 1)
        
        # Test first with predicate
        self.assertEqual(nums.first(lambda x: x > 3), 4)
        
        # Test first with no match
        with self.assertRaises(RuntimeError):
            nums.first(lambda x: x > 10)
        
        # Test first on empty sequence
        with self.assertRaises(IndexError):
            PyLinq().first()
        
        # Test first_or_default with no predicate
        self.assertEqual(nums.first_or_default(0), 1)
        
        # Test first_or_default with predicate
        self.assertEqual(nums.first_or_default(0, lambda x: x > 3), 4)
        
        # Test first_or_default with no match
        self.assertEqual(nums.first_or_default(0, lambda x: x > 10), 0)
        
        # Test first_or_default on empty sequence
        self.assertEqual(PyLinq().first_or_default(0), 0)

    def test_for_each(self):
        nums = PyLinq([1, 2, 3, 4, 5])
        
        # Test for_each
        result = []
        nums.for_each(lambda x: result.append(x * 2))
        self.assertEqual(result, [2, 4, 6, 8, 10])

        my_func = mock.MagicMock(name="mock_func")
        nums.for_each(my_func)
        self.assertEqual(my_func.call_count, 5)

    def test_for_each_concurrent(self):
        nums = PyLinq(list(range(10)))
        
        # Test for_each_concurrent
        results = []
        lock = threading.Lock()
        
        def process(x):
            time.sleep(0.01)  # Small delay to ensure parallel execution
            with lock:
                results.append(x * 2)
        
        nums.for_each_concurrent(process, max_workers=4)
        
        # The order may be different due to for_each_concurrent execution
        self.assertEqual(sorted(results), [0, 2, 4, 6, 8, 10, 12, 14, 16, 18])

        my_func = mock.MagicMock(name="mock_func")
        nums.for_each_concurrent(my_func)
        self.assertEqual(my_func.call_count, 10)

    def test_group_by(self):
        people = PyLinq([
            {"name": "Alice", "dept": "IT"},
            {"name": "Bob", "dept": "HR"},
            {"name": "Charlie", "dept": "IT"},
            {"name": "David", "dept": "Finance"},
            {"name": "Eve", "dept": "HR"}
        ])
        
        # Test group_by
        grouped = people.group_by(lambda p: p["dept"])
        
        # Convert to dict for easier testing
        grouped_dict = {key: group.to_list() for key, group in grouped}
        
        self.assertEqual(len(grouped_dict), 3)
        self.assertEqual(len(grouped_dict["IT"]), 2)
        self.assertEqual(len(grouped_dict["HR"]), 2)
        self.assertEqual(len(grouped_dict["Finance"]), 1)

    def test_inner_join(self):
        employees = PyLinq([
            {"id": 1, "name": "Alice", "dept_id": 101},
            {"id": 2, "name": "Bob", "dept_id": 102},
            {"id": 3, "name": "Charlie", "dept_id": 101},
            {"id": 4, "name": "David", "dept_id": 103}
        ])
        
        departments = [
            {"id": 101, "name": "IT"},
            {"id": 102, "name": "HR"},
            {"id": 103, "name": "Finance"}
        ]
        
        # Test inner_join
        result = employees.inner_join(
            departments,
            lambda e: e["dept_id"],
            lambda d: d["id"],
            lambda e, d: {"name": e["name"], "dept": d["name"]}
        )
        
        expected = [
            {"name": "Alice", "dept": "IT"},
            {"name": "Bob", "dept": "HR"},
            {"name": "Charlie", "dept": "IT"},
            {"name": "David", "dept": "Finance"}
        ]
        
        self.assertEqual(result.to_list(), expected)

    def test_intersect(self):
        nums1 = PyLinq([1, 2, 3, 4, 5])
        nums2 = [3, 4, 5, 6, 7]
        
        # Test intersect
        result = nums1.intersect(nums2)
        self.assertEqual(sorted(result.to_list()), [3, 4, 5])

    def test_join_string(self):
        # Test join with strings
        words = PyLinq(["Hello", "world", "from", "PyLinq"])
        self.assertEqual(words.join(","), "Hello,world,from,PyLinq")
        
        # Test join with numbers
        nums = PyLinq([1, 2, 3, 4, 5])
        self.assertEqual(nums.join(", "), "1, 2, 3, 4, 5")

    def test_last_and_last_or_default(self):
        nums = PyLinq([1, 2, 3, 4, 5])
        
        # Test last with no predicate
        self.assertEqual(nums.last(), 5)
        
        # Test last with predicate
        self.assertEqual(nums.last(lambda x: x < 4), 3)
        
        # Test last with no match
        with self.assertRaises(ValueError):
            nums.last(lambda x: x > 10)
        
        # Test last on empty sequence
        with self.assertRaises(ValueError):
            PyLinq().last()
        
        # Test last_or_default with no predicate
        self.assertEqual(nums.last_or_default(0), 5)
        
        # Test last_or_default with predicate
        self.assertEqual(nums.last_or_default(0, lambda x: x < 4), 3)
        
        # Test last_or_default with no match
        self.assertEqual(nums.last_or_default(0, lambda x: x > 10), 0)
        
        # Test last_or_default on empty sequence
        self.assertEqual(PyLinq().last_or_default(0), 0)

    def test_map_and_select(self):
        nums = PyLinq([1, 2, 3, 4, 5])
        
        # Test map
        self.assertEqual(nums.map(lambda x: x * 2).to_list(), [2, 4, 6, 8, 10])
        
        # Test select (alias for map)
        self.assertEqual(nums.select(lambda x: x * 2).to_list(), [2, 4, 6, 8, 10])

    def test_max_and_min(self):
        nums = PyLinq([3, 1, 4, 1, 5, 9, 2, 6])
        
        # Test max with no selector
        self.assertEqual(nums.max(), 9)
        
        # Test max with selector
        people = PyLinq([
            {"name": "Alice", "age": 25},
            {"name": "Bob", "age": 30},
            {"name": "Charlie", "age": 35}
        ])
        self.assertEqual(people.max(lambda p: p["age"]), {"name": "Charlie", "age": 35})
        
        # Test min with no selector
        self.assertEqual(nums.min(), 1)
        
        # Test min with selector
        self.assertEqual(people.min(lambda p: p["age"]), {"name": "Alice", "age": 25})
        
        # Test empty sequence
        with self.assertRaises(ValueError):
            PyLinq().max()
        with self.assertRaises(ValueError):
            PyLinq().min()

    def test_order_by_and_sort(self):
        nums = PyLinq([3, 1, 4, 1, 5, 9, 2, 6])
        
        # Test order_by with no selector
        self.assertEqual(nums.order_by().to_list(), [1, 1, 2, 3, 4, 5, 6, 9])
        
        # Test order_by with selector
        people = PyLinq([
            {"name": "Charlie", "age": 35},
            {"name": "Alice", "age": 25},
            {"name": "Bob", "age": 30}
        ])
        
        sorted_by_name = people.order_by(lambda p: p["name"]).to_list()
        self.assertEqual(sorted_by_name[0]["name"], "Alice")
        self.assertEqual(sorted_by_name[1]["name"], "Bob")
        self.assertEqual(sorted_by_name[2]["name"], "Charlie")
        
        # Test sort (alias for order_by)
        self.assertEqual(nums.sort().to_list(), [1, 1, 2, 3, 4, 5, 6, 9])
        
        # Test order_by_descending
        self.assertEqual(nums.order_by_descending().to_list(), [9, 6, 5, 4, 3, 2, 1, 1])
        
        sorted_by_age_desc = people.order_by_descending(lambda p: p["age"]).to_list()
        self.assertEqual(sorted_by_age_desc[0]["name"], "Charlie")
        self.assertEqual(sorted_by_age_desc[1]["name"], "Bob")
        self.assertEqual(sorted_by_age_desc[2]["name"], "Alice")

    def test_sequence_equal(self):
        nums1 = PyLinq([1, 2, 3, 4, 5])
        nums2 = [1, 2, 3, 4, 5]
        nums3 = [1, 2, 3, 4]
        nums4 = [5, 4, 3, 2, 1]
        
        # Test sequence_equal
        self.assertTrue(nums1.sequence_equal(nums2))
        self.assertFalse(nums1.sequence_equal(nums3))
        self.assertFalse(nums1.sequence_equal(nums4))

    def test_single_and_single_or_default(self):
        # Test single with no predicate
        single_item = PyLinq([42])
        self.assertEqual(single_item.single(), 42)
        
        # Test single with predicate
        multiple_items = PyLinq([1, 2, 3, 4, 5])
        self.assertEqual(multiple_items.single(lambda x: x == 3), 3)
        
        # Test single with no match
        with self.assertRaises(ValueError):
            multiple_items.single(lambda x: x > 10)
        
        # Test single with multiple matches
        with self.assertRaises(ValueError):
            multiple_items.single(lambda x: x > 3)
        
        # Test single on empty sequence
        with self.assertRaises(ValueError):
            PyLinq().single()
        
        # Test single on sequence with multiple items and no predicate
        with self.assertRaises(ValueError):
            multiple_items.single()
        
        # Test single_or_default with no predicate
        self.assertEqual(single_item.single_or_default(0), 42)
        
        # Test single_or_default with predicate
        self.assertEqual(multiple_items.single_or_default(0, lambda x: x == 3), 3)
        
        # Test single_or_default with no match
        self.assertEqual(multiple_items.single_or_default(0, lambda x: x > 10), 0)
        
        # Test single_or_default with multiple matches
        self.assertEqual(multiple_items.single_or_default(0, lambda x: x > 3), 0)
        
        # Test single_or_default on empty sequence
        self.assertEqual(PyLinq().single_or_default(0), 0)

    def test_skip_and_skip_while(self):
        nums = PyLinq([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # Test skip
        self.assertEqual(nums.skip(3).to_list(), [4, 5, 6, 7, 8, 9, 10])
        self.assertEqual(nums.skip(0).to_list(), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertEqual(nums.skip(20).to_list(), [])
        
        # Test skip_while
        self.assertEqual(nums.skip_while(lambda x: x < 5).to_list(), [5, 6, 7, 8, 9, 10])
        self.assertEqual(nums.skip_while(lambda x: x > 100).to_list(), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertEqual(nums.skip_while(lambda x: True).to_list(), [])

    def test_sum(self):
        nums = PyLinq([1, 2, 3, 4, 5])
        
        # Test sum with no selector
        self.assertEqual(nums.sum(), 15)
        
        # Test sum with selector
        people = PyLinq([
            {"name": "Alice", "age": 25},
            {"name": "Bob", "age": 30},
            {"name": "Charlie", "age": 35}
        ])
        self.assertEqual(people.sum(lambda p: p["age"]), 90)
        
        # Test sum on empty sequence
        self.assertEqual(PyLinq().sum(), 0)

    def test_take_and_take_while(self):
        nums = PyLinq([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # Test take
        self.assertEqual(nums.take(3).to_list(), [1, 2, 3])
        self.assertEqual(nums.take(0).to_list(), [])
        self.assertEqual(nums.take(20).to_list(), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # Test take_while
        self.assertEqual(nums.take_while(lambda x: x < 5).to_list(), [1, 2, 3, 4])
        self.assertEqual(nums.take_while(lambda x: x > 100).to_list(), [])
        self.assertEqual(nums.take_while(lambda x: True).to_list(), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    def test_to_collection_conversions(self):
        nums = PyLinq([3, 1, 4, 1, 5, 9, 2, 6])
        
        # Test to_list
        self.assertEqual(nums.to_list(), [3, 1, 4, 1, 5, 9, 2, 6])
        
        # Test to_tuple
        self.assertEqual(nums.to_tuple(), (3, 1, 4, 1, 5, 9, 2, 6))
        
        # Test to_set
        self.assertEqual(nums.to_set(), {1, 2, 3, 4, 5, 6, 9})
        
        # Test to_dict
        people = PyLinq([
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
            {"id": 3, "name": "Charlie"}
        ])
        
        # With key selector only
        result_dict = people.to_dict(lambda p: p["id"])
        expected_dict = {
            1: {'id': 1, 'name': 'Alice'}, 
            2: {'id': 2, 'name': 'Bob'}, 
            3: {'id': 3, 'name': 'Charlie'}
        }
        self.assertDictEqual(result_dict, expected_dict)

        
        # With key and value selectors
        name_dict = people.to_dict(lambda p: p["id"], lambda p: p["name"])
        self.assertEqual(name_dict, {1: "Alice", 2: "Bob", 3: "Charlie"})

    def test_union(self):
        nums1 = PyLinq([1, 2, 3, 3, 4, 5])
        nums2 = [3, 4, 5, 6, 7]
        
        # Test union
        result = nums1.union(nums2)
        self.assertEqual(result.to_set(), {1, 2, 3, 4, 5, 6, 7})

    def test_zip(self):
        nums = PyLinq([1, 2, 3, 4, 5])
        letters = ["a", "b", "c", "d", "e"]
        
        # Test zip
        result = nums.zip(letters)
        expected = [(1, "a"), (2, "b"), (3, "c"), (4, "d"), (5, "e")]
        self.assertEqual(result.to_list(), expected)
        
        # Test zip with sequences of different lengths
        short_seq = [10, 20, 30]
        result = nums.zip(short_seq)
        expected = [(1, 10), (2, 20), (3, 30)]
        self.assertEqual(result.to_list(), expected)
