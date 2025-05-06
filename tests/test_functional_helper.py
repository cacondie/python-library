from unittest import TestCase, mock
from src.functional_helpers import for_each, filter_map, for_each_parallel


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

    def test_for_each_parallel(self):
        my_list = [1, 2, 3, 4, 5, 6]
        my_func = mock.MagicMock(name="mock_func")
        for_each_parallel(my_func, my_list)

        self.assertEqual(my_func.call_count, 6)