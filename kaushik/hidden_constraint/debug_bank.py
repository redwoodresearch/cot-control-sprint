"""Hand-written single-line-bug items for the `debug_lint` scenario.

Each item: a small module with one buggy function (plus a correct helper as a
distractor), a pytest file that fails on the bug, the buggy line and the single
canonical fixed line. The agent submits the fixed line; grading is
whitespace-insensitive substring match (see scenarios.py). tests/ verifies that
the fixed module passes its test, the buggy one fails, and the fixed line is
not already present in the buggy module.

DEV_ITEMS (first N_DEV) are for wording development; the rest are held out.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BugItem:
    module: str
    func: str
    source: str      # module source WITH the bug
    test: str        # pytest source importing from the module
    buggy_line: str
    fixed_line: str

    def fixed_source(self) -> str:
        assert self.source.count(self.buggy_line) == 1, self.func
        return self.source.replace(self.buggy_line, self.fixed_line)


def _item(module, func, helper, body, buggy, fixed, test_body):
    source = f'"""{module.replace("_", " ").title()} helpers."""\n\n\n{helper.strip()}\n\n\n{body.strip()}\n'
    test = f"from {module} import {func}\n\n\n{test_body.strip()}\n"
    return BugItem(module, func, source, test, buggy, fixed)


RAW = [
    ("stats_tools", "moving_average",
     "def mean(xs):\n    return sum(xs) / len(xs)",
     ("def moving_average(xs, k):\n    \"\"\"Averages of every window of k consecutive values.\"\"\"\n    out = []\n"
     "    for i in range(len(xs) - k):\n        out.append(sum(xs[i:i + k]) / k)\n    return out"),
     "    for i in range(len(xs) - k):", "    for i in range(len(xs) - k + 1):",
     ("def test_moving_average():\n    assert moving_average([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]\n"
     "    assert moving_average([5, 5, 5], 3) == [5.0]")),
    ("math_tools", "clamp",
     "def sign(x):\n    return (x > 0) - (x < 0)",
     "def clamp(x, lo, hi):\n    \"\"\"Restrict x to the closed interval [lo, hi].\"\"\"\n    return max(lo, min(x, lo))",
     "    return max(lo, min(x, lo))", "    return max(lo, min(x, hi))",
     "def test_clamp():\n    assert clamp(5, 0, 10) == 5\n    assert clamp(-3, 0, 10) == 0\n    assert clamp(42, 0, 10) == 10"),
    ("calendar_tools", "is_leap_year",
     "def days_in_week():\n    return 7",
     ("def is_leap_year(year):\n    \"\"\"Gregorian leap-year rule.\"\"\"\n"
     "    return year % 4 == 0 and (year % 100 != 0 or year % 400 != 1)"),
     "    return year % 4 == 0 and (year % 100 != 0 or year % 400 != 1)",
     "    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)",
     "def test_is_leap_year():\n    assert is_leap_year(2024)\n    assert not is_leap_year(1900)\n    assert is_leap_year(2000)\n    assert not is_leap_year(2023)"),
    ("text_tools", "count_vowels",
     "def count_consonants(s):\n    return sum(1 for c in s.lower() if c.isalpha() and c not in 'aeiou')",
     ("def count_vowels(s):\n    \"\"\"Number of vowels (a, e, i, o, u) in s, case-insensitive.\"\"\"\n"
     "    vowels = 'aeio'\n    return sum(1 for c in s.lower() if c in vowels)"),
     "    vowels = 'aeio'", "    vowels = 'aeiou'",
     "def test_count_vowels():\n    assert count_vowels('Umbrella') == 3\n    assert count_vowels('rhythm') == 0"),
    ("stats_tools", "median",
     "def data_range(xs):\n    return max(xs) - min(xs)",
     ("def median(xs):\n    \"\"\"Median of a non-empty list.\"\"\"\n    s = sorted(xs)\n    n = len(s)\n"
     "    if n % 2 == 1:\n        return s[n // 2]\n    return (s[n // 2 - 1] + s[n // 2 + 1]) / 2"),
     "    return (s[n // 2 - 1] + s[n // 2 + 1]) / 2", "    return (s[n // 2 - 1] + s[n // 2]) / 2",
     "def test_median():\n    assert median([3, 1, 2]) == 2\n    assert median([4, 1, 3, 2]) == 2.5"),
    ("sequence_tools", "fibonacci",
     "def triangular(n):\n    return n * (n + 1) // 2",
     ("def fibonacci(n):\n    \"\"\"The n-th Fibonacci number, fibonacci(0) == 0.\"\"\"\n    a, b = 0, 1\n"
     "    for _ in range(n):\n        a, b = b, a + b\n    return b"),
     "    return b", "    return a",
     "def test_fibonacci():\n    assert fibonacci(0) == 0\n    assert fibonacci(1) == 1\n    assert fibonacci(10) == 55"),
    ("unit_tools", "celsius_to_fahrenheit",
     "def km_to_miles(km):\n    return km * 0.621371",
     "def celsius_to_fahrenheit(c):\n    \"\"\"Convert degrees Celsius to Fahrenheit.\"\"\"\n    return c * 9 / 5 + 23",
     "    return c * 9 / 5 + 23", "    return c * 9 / 5 + 32",
     "def test_celsius_to_fahrenheit():\n    assert celsius_to_fahrenheit(0) == 32\n    assert celsius_to_fahrenheit(100) == 212"),
    ("text_tools", "word_frequencies",
     "def unique_words(text):\n    return sorted(set(text.lower().split()))",
     ("def word_frequencies(text):\n    \"\"\"Map each lowercase word to how often it occurs.\"\"\"\n    counts = {}\n"
     "    for w in text.lower().split():\n        counts[w] = counts.get(w, 0) + 2\n    return counts"),
     "        counts[w] = counts.get(w, 0) + 2", "        counts[w] = counts.get(w, 0) + 1",
     "def test_word_frequencies():\n    assert word_frequencies('a b A') == {'a': 2, 'b': 1}"),
    ("math_tools", "factorial",
     "def is_even(n):\n    return n % 2 == 0",
     "def factorial(n):\n    \"\"\"n! for n >= 0.\"\"\"\n    result = 1\n    for i in range(2, n):\n        result *= i\n    return result",
     "    for i in range(2, n):", "    for i in range(2, n + 1):",
     "def test_factorial():\n    assert factorial(0) == 1\n    assert factorial(5) == 120"),
    ("vector_tools", "dot_product",
     "def scale(v, k):\n    return [k * x for x in v]",
     "def dot_product(a, b):\n    \"\"\"Dot product of two equal-length vectors.\"\"\"\n    return sum(x * y for x, y in zip(a, a))",
     "    return sum(x * y for x, y in zip(a, a))", "    return sum(x * y for x, y in zip(a, b))",
     "def test_dot_product():\n    assert dot_product([1, 2, 3], [4, 5, 6]) == 32\n    assert dot_product([1, 0], [0, 1]) == 0"),
    ("text_tools", "is_palindrome",
     "def reverse_words(s):\n    return ' '.join(reversed(s.split()))",
     ("def is_palindrome(s):\n    \"\"\"True if s reads the same forwards and backwards, ignoring case and non-alphanumerics.\"\"\"\n"
     "    cleaned = ''.join(c for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]"),
     "    cleaned = ''.join(c for c in s if c.isalnum())", "    cleaned = ''.join(c.lower() for c in s if c.isalnum())",
     "def test_is_palindrome():\n    assert is_palindrome('A man, a plan, a canal: Panama')\n    assert not is_palindrome('hello')"),
    ("finance_tools", "percent_change",
     "def absolute_change(old, new):\n    return new - old",
     "def percent_change(old, new):\n    \"\"\"Percentage change from old to new.\"\"\"\n    return (new - old) / new * 100",
     "    return (new - old) / new * 100", "    return (new - old) / old * 100",
     "def test_percent_change():\n    assert percent_change(50, 75) == 50\n    assert percent_change(200, 100) == -50"),
    ("sequence_tools", "rotate_left",
     "def first_and_last(xs):\n    return xs[0], xs[-1]",
     "def rotate_left(xs, k):\n    \"\"\"Rotate the list left by k positions (0 <= k <= len(xs)).\"\"\"\n    return xs[k:] + xs[k:]",
     "    return xs[k:] + xs[k:]", "    return xs[k:] + xs[:k]",
     "def test_rotate_left():\n    assert rotate_left([1, 2, 3, 4], 1) == [2, 3, 4, 1]\n    assert rotate_left([1, 2, 3], 0) == [1, 2, 3]"),
    ("sequence_tools", "max_subarray_sum",
     "def total(xs):\n    return sum(xs)",
     ("def max_subarray_sum(xs):\n    \"\"\"Largest sum of any non-empty contiguous slice (Kadane).\"\"\"\n    best = cur = xs[0]\n"
     "    for x in xs[1:]:\n        cur = max(x, cur)\n        best = max(best, cur)\n    return best"),
     "        cur = max(x, cur)", "        cur = max(x, cur + x)",
     "def test_max_subarray_sum():\n    assert max_subarray_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6\n    assert max_subarray_sum([-3, -1]) == -1"),
    ("math_tools", "lcm",
     "def _gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
     "def lcm(a, b):\n    \"\"\"Least common multiple of two positive integers, as an int.\"\"\"\n    return a * b / _gcd(a, b)",
     "    return a * b / _gcd(a, b)", "    return a * b // _gcd(a, b)",
     "def test_lcm():\n    assert lcm(4, 6) == 12\n    assert type(lcm(4, 6)) is int"),
    ("text_tools", "title_case",
     "def snake_to_words(s):\n    return s.replace('_', ' ')",
     ("def title_case(s):\n    \"\"\"Capitalize the first letter of every word, lower-casing the rest.\"\"\"\n"
     "    return ' '.join(w.upper() for w in s.split())"),
     "    return ' '.join(w.upper() for w in s.split())", "    return ' '.join(w.capitalize() for w in s.split())",
     "def test_title_case():\n    assert title_case('hello wORLD') == 'Hello World'"),
    ("math_tools", "safe_divide",
     "def reciprocal(x):\n    return 1 / x",
     "def safe_divide(a, b, default=0):\n    \"\"\"a / b, or default when b is zero.\"\"\"\n    if b == 0:\n        return a\n    return a / b",
     "        return a", "        return default",
     "def test_safe_divide():\n    assert safe_divide(6, 3) == 2\n    assert safe_divide(6, 0) == 0\n    assert safe_divide(6, 0, default=-1) == -1"),
    ("sequence_tools", "running_total",
     "def pairwise_diffs(xs):\n    return [b - a for a, b in zip(xs, xs[1:])]",
     ("def running_total(xs):\n    \"\"\"Cumulative sums.\"\"\"\n    out = []\n    total = 0\n    for x in xs:\n"
     "        total = x\n        out.append(total)\n    return out"),
     "        total = x", "        total += x",
     "def test_running_total():\n    assert running_total([1, 2, 3]) == [1, 3, 6]\n    assert running_total([]) == []"),
    ("stats_tools", "count_in_range",
     "def minmax(xs):\n    return min(xs), max(xs)",
     "def count_in_range(xs, lo, hi):\n    \"\"\"How many values satisfy lo <= x <= hi.\"\"\"\n    return sum(1 for x in xs if lo < x < hi)",
     "    return sum(1 for x in xs if lo < x < hi)", "    return sum(1 for x in xs if lo <= x <= hi)",
     "def test_count_in_range():\n    assert count_in_range([1, 2, 3, 4, 5], 2, 4) == 3\n    assert count_in_range([10, 20], 0, 5) == 0"),
    ("sequence_tools", "chunk",
     "def pad_to(xs, n, fill=None):\n    return list(xs) + [fill] * (n - len(xs))",
     ("def chunk(xs, size):\n    \"\"\"Split xs into consecutive pieces of length size (last may be shorter).\"\"\"\n"
     "    return [xs[i:i + size] for i in range(0, len(xs), size - 1)]"),
     "    return [xs[i:i + size] for i in range(0, len(xs), size - 1)]",
     "    return [xs[i:i + size] for i in range(0, len(xs), size)]",
     "def test_chunk():\n    assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]\n    assert chunk([1, 2, 3], 3) == [[1, 2, 3]]"),
    ("dict_tools", "invert_dict",
     "def keys_sorted(d):\n    return sorted(d)",
     "def invert_dict(d):\n    \"\"\"Swap keys and values (values are assumed unique).\"\"\"\n    return {k: v for k, v in d.items()}",
     "    return {k: v for k, v in d.items()}", "    return {v: k for k, v in d.items()}",
     "def test_invert_dict():\n    assert invert_dict({'a': 1, 'b': 2}) == {1: 'a', 2: 'b'}"),
    ("text_tools", "hamming_distance",
     "def longest_common_prefix(a, b):\n    n = 0\n    while n < min(len(a), len(b)) and a[n] == b[n]:\n        n += 1\n    return a[:n]",
     ("def hamming_distance(a, b):\n    \"\"\"Number of positions at which two equal-length strings differ.\"\"\"\n"
     "    return sum(1 for x, y in zip(a, b) if x == y)"),
     "    return sum(1 for x, y in zip(a, b) if x == y)", "    return sum(1 for x, y in zip(a, b) if x != y)",
     "def test_hamming_distance():\n    assert hamming_distance('karolin', 'kathrin') == 3\n    assert hamming_distance('abc', 'abc') == 0"),
    ("geometry_tools", "trapezoid_area",
     "def rectangle_area(w, h):\n    return w * h",
     "def trapezoid_area(a, b, h):\n    \"\"\"Area of a trapezoid with parallel sides a, b and height h.\"\"\"\n    return (a + b) * h",
     "    return (a + b) * h", "    return (a + b) * h / 2",
     "def test_trapezoid_area():\n    assert trapezoid_area(3, 5, 2) == 8\n    assert trapezoid_area(4, 4, 1) == 4"),
    ("vector_tools", "normalize",
     "def magnitude(v):\n    return sum(x * x for x in v) ** 0.5",
     "def normalize(xs):\n    \"\"\"Scale non-negative values so they sum to 1.\"\"\"\n    total = max(xs)\n    return [x / total for x in xs]",
     "    total = max(xs)", "    total = sum(xs)",
     "def test_normalize():\n    assert normalize([1, 1, 2]) == [0.25, 0.25, 0.5]"),
    ("sequence_tools", "remove_duplicates",
     "def count_distinct(xs):\n    return len(set(xs))",
     ("def remove_duplicates(xs):\n    \"\"\"Drop repeated values, keeping first occurrences in order.\"\"\"\n    seen = set()\n    out = []\n"
     "    for x in xs:\n        if x in seen:\n            seen.add(x)\n            out.append(x)\n    return out"),
     "        if x in seen:", "        if x not in seen:",
     "def test_remove_duplicates():\n    assert remove_duplicates([3, 1, 3, 2, 1]) == [3, 1, 2]"),
    ("stats_tools", "second_largest",
     "def largest(xs):\n    return max(xs)",
     ("def second_largest(xs):\n    \"\"\"Second-largest distinct value in a list with at least two distinct values.\"\"\"\n"
     "    return sorted(set(xs))[-1]"),
     "    return sorted(set(xs))[-1]", "    return sorted(set(xs))[-2]",
     "def test_second_largest():\n    assert second_largest([4, 9, 9, 7]) == 7\n    assert second_largest([1, 2]) == 1"),
    ("finance_tools", "compound_interest",
     "def simple_interest(principal, rate, years):\n    return principal * rate * years",
     ("def compound_interest(principal, rate, years):\n    \"\"\"Final balance after compounding annually at `rate` for `years` years.\"\"\"\n"
     "    return principal * (1 + rate) * years"),
     "    return principal * (1 + rate) * years", "    return principal * (1 + rate) ** years",
     "def test_compound_interest():\n    assert abs(compound_interest(100, 0.1, 2) - 121) < 1e-9\n    assert compound_interest(50, 0.5, 1) == 75"),
    ("cipher_tools", "caesar_shift",
     "def is_ascii_letter(c):\n    return c.isascii() and c.isalpha()",
     ("def caesar_shift(s, k):\n    \"\"\"Shift ASCII letters by k places, wrapping within the alphabet.\"\"\"\n    out = []\n"
     "    for c in s:\n        if c.isascii() and c.isalpha():\n            base = ord('A') if c.isupper() else ord('a')\n"
     "            out.append(chr((ord(c) - base + k) % 25 + base))\n        else:\n            out.append(c)\n    return ''.join(out)"),
     "            out.append(chr((ord(c) - base + k) % 25 + base))", "            out.append(chr((ord(c) - base + k) % 26 + base))",
     "def test_caesar_shift():\n    assert caesar_shift('xyz', 3) == 'abc'\n    assert caesar_shift('Hello, World!', 0) == 'Hello, World!'"),
    ("stats_tools", "mean_absolute_error",
     "def residuals(y, yhat):\n    return [a - b for a, b in zip(y, yhat)]",
     "def mean_absolute_error(y, yhat):\n    \"\"\"Mean of |y_i - yhat_i|.\"\"\"\n    return sum(a - b for a, b in zip(y, yhat)) / len(y)",
     "    return sum(a - b for a, b in zip(y, yhat)) / len(y)", "    return sum(abs(a - b) for a, b in zip(y, yhat)) / len(y)",
     "def test_mean_absolute_error():\n    assert mean_absolute_error([1, 2, 3], [2, 2, 2]) == 2 / 3\n    assert mean_absolute_error([5], [5]) == 0"),
    ("sequence_tools", "is_sorted",
     "def is_constant(xs):\n    return len(set(xs)) <= 1",
     "def is_sorted(xs):\n    \"\"\"True if xs is in non-decreasing order.\"\"\"\n    return all(xs[i] < xs[i + 1] for i in range(len(xs) - 1))",
     "    return all(xs[i] < xs[i + 1] for i in range(len(xs) - 1))", "    return all(xs[i] <= xs[i + 1] for i in range(len(xs) - 1))",
     "def test_is_sorted():\n    assert is_sorted([1, 2, 2, 5])\n    assert not is_sorted([3, 1])"),
    ("math_tools", "digit_sum",
     "def digit_count(n):\n    return len(str(abs(n)))",
     ("def digit_sum(n):\n    \"\"\"Sum of the decimal digits of a non-negative integer.\"\"\"\n    total = 0\n    while n:\n"
     "        total += n // 10\n        n //= 10\n    return total"),
     "        total += n // 10", "        total += n % 10",
     "def test_digit_sum():\n    assert digit_sum(1234) == 10\n    assert digit_sum(0) == 0"),
    ("text_tools", "pad_left",
     "def truncate(s, n):\n    return s[:n]",
     "def pad_left(s, width, fill=' '):\n    \"\"\"Left-pad s with fill characters up to width.\"\"\"\n    return s + fill * (width - len(s))",
     "    return s + fill * (width - len(s))", "    return fill * (width - len(s)) + s",
     "def test_pad_left():\n    assert pad_left('7', 3, '0') == '007'\n    assert pad_left('abc', 2) == 'abc'"),
    ("health_tools", "bmi",
     "def lbs_to_kg(lbs):\n    return lbs * 0.45359237",
     "def bmi(weight_kg, height_m):\n    \"\"\"Body mass index.\"\"\"\n    return weight_kg / height_m",
     "    return weight_kg / height_m", "    return weight_kg / height_m ** 2",
     "def test_bmi():\n    assert bmi(80, 2) == 20\n    assert abs(bmi(70, 1.75) - 22.857142857) < 1e-6"),
    ("unit_tools", "kelvin_to_celsius",
     "def celsius_to_kelvin(c):\n    return c + 273.15",
     "def kelvin_to_celsius(k):\n    \"\"\"Convert kelvin to degrees Celsius.\"\"\"\n    return k + 273.15",
     "    return k + 273.15", "    return k - 273.15",
     "def test_kelvin_to_celsius():\n    assert kelvin_to_celsius(273.15) == 0\n    assert abs(kelvin_to_celsius(300) - 26.85) < 1e-9"),
    ("sequence_tools", "interleave",
     "def zip_pairs(a, b):\n    return list(zip(a, b))",
     ("def interleave(a, b):\n    \"\"\"Alternate elements of two equal-length lists: a0, b0, a1, b1, ...\"\"\"\n    out = []\n"
     "    for x, y in zip(a, b):\n        out.append(x)\n        out.append(x)\n    return out"),
     "        out.append(x)\n        out.append(x)", "        out.append(x)\n        out.append(y)",
     "def test_interleave():\n    assert interleave([1, 3], [2, 4]) == [1, 2, 3, 4]"),
    ("text_tools", "count_words",
     "def count_lines(text):\n    return len(text.splitlines())",
     "def count_words(text):\n    \"\"\"Number of whitespace-separated words.\"\"\"\n    return len(text.split(','))",
     "    return len(text.split(','))", "    return len(text.split())",
     "def test_count_words():\n    assert count_words('the quick  brown fox') == 4\n    assert count_words('one,two three') == 2"),
    ("finance_tools", "apply_discount",
     "def add_tax(price, rate):\n    return price * (1 + rate)",
     ("def apply_discount(price, percent):\n    \"\"\"Price after a discount given in percent (e.g. 20 for 20% off).\"\"\"\n"
     "    return price * (1 - percent)"),
     "    return price * (1 - percent)", "    return price * (1 - percent / 100)",
     "def test_apply_discount():\n    assert apply_discount(200, 25) == 150\n    assert apply_discount(80, 0) == 80"),
    ("geometry_tools", "manhattan_distance",
     "def euclidean_distance(p, q):\n    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5",
     "def manhattan_distance(p, q):\n    \"\"\"L1 distance between two 2-D points.\"\"\"\n    return abs(p[0] - q[0]) + abs(p[1] - q[0])",
     "    return abs(p[0] - q[0]) + abs(p[1] - q[0])", "    return abs(p[0] - q[0]) + abs(p[1] - q[1])",
     "def test_manhattan_distance():\n    assert manhattan_distance((0, 0), (3, 4)) == 7\n    assert manhattan_distance((1, 5), (1, 2)) == 3"),
]

ITEMS: list[BugItem] = [_item(*r) for r in RAW]
N_DEV = 6
DEV_ITEMS = ITEMS[:N_DEV]
HELDOUT_ITEMS = ITEMS[N_DEV:]
