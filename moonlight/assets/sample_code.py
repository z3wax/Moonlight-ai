"""A small collection of classic algorithms and data structures in idiomatic Python.

This module is bundled with Moonlight as sample code for tokenizer training, perplexity
benchmarks and offline smoke tests. Every function is self-contained, uses only the standard
library and is covered by ``_self_test``. Run the module directly to see a short demo::

    python sample_code.py
"""

from __future__ import annotations

import heapq
import math
import re
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
from typing import Dict, Generic, Hashable, List, Optional, Sequence, Tuple, TypeVar

T = TypeVar("T")
K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


# ---------------------------------------------------------------------------
# Number theory
# ---------------------------------------------------------------------------

def gcd(a: int, b: int) -> int:
    """Return the greatest common divisor of ``a`` and ``b`` using Euclid's algorithm."""
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def lcm(a: int, b: int) -> int:
    """Return the least common multiple of ``a`` and ``b`` (0 if either is 0)."""
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(a, b)


def is_prime(n: int) -> bool:
    """Return True if ``n`` is a prime number (trial division by 6k +/- 1)."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def primes_up_to(limit: int) -> List[int]:
    """Return all primes ``<= limit`` with the sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, math.isqrt(limit) + 1):
        if sieve[p]:
            sieve[p * p::p] = bytearray(len(range(p * p, limit + 1, p)))
    return [i for i, flag in enumerate(sieve) if flag]


def prime_factors(n: int) -> List[int]:
    """Return the prime factorization of ``n >= 2`` in non-decreasing order."""
    if n < 2:
        raise ValueError("n must be at least 2")
    factors: List[int] = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        factors.append(n)
    return factors


def fibonacci(n: int) -> int:
    """Return the n-th Fibonacci number (F(0) = 0, F(1) = 1) iteratively."""
    if n < 0:
        raise ValueError("n must be non-negative")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


# ---------------------------------------------------------------------------
# Searching and sorting
# ---------------------------------------------------------------------------

def binary_search(items: Sequence[int], target: int) -> int:
    """Return the index of ``target`` in the sorted sequence ``items``, or -1 if absent."""
    lo, hi = 0, len(items) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if items[mid] == target:
            return mid
        if items[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def merge_sort(items: Sequence[T]) -> List[T]:
    """Return a new sorted list; stable, O(n log n) time."""
    if len(items) <= 1:
        return list(items)
    mid = len(items) // 2
    left, right = merge_sort(items[:mid]), merge_sort(items[mid:])
    merged: List[T] = []
    i = j = 0
    while i < len(left) and j < len(right):
        if right[j] < left[i]:  # strict comparison keeps equal elements in order
            merged.append(right[j])
            j += 1
        else:
            merged.append(left[i])
            i += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged


def quicksort(items: Sequence[T]) -> List[T]:
    """Return a new sorted list using a three-way partition around the middle element."""
    if len(items) <= 1:
        return list(items)
    pivot = items[len(items) // 2]
    smaller = [x for x in items if x < pivot]
    equal = [x for x in items if x == pivot]
    larger = [x for x in items if pivot < x]
    return quicksort(smaller) + equal + quicksort(larger)


# ---------------------------------------------------------------------------
# Dynamic programming
# ---------------------------------------------------------------------------

def edit_distance(a: str, b: str) -> int:
    """Return the Levenshtein distance between ``a`` and ``b`` using O(len(b)) memory."""
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,               # deletion
                current[j - 1] + 1,            # insertion
                previous[j - 1] + (ca != cb),  # substitution
            ))
        previous = current
    return previous[-1]


def knapsack(weights: Sequence[int], values: Sequence[int], capacity: int) -> int:
    """Return the best total value of a 0/1 knapsack with the given ``capacity``."""
    if len(weights) != len(values):
        raise ValueError("weights and values must have the same length")
    best = [0] * (capacity + 1)
    for weight, value in zip(weights, values):
        for c in range(capacity, weight - 1, -1):
            best[c] = max(best[c], best[c - weight] + value)
    return best[capacity]


# ---------------------------------------------------------------------------
# Strings
# ---------------------------------------------------------------------------

def is_palindrome(text: str) -> bool:
    """Return True if ``text`` reads the same backwards, ignoring case and punctuation."""
    letters = [ch.lower() for ch in text if ch.isalnum()]
    return letters == letters[::-1]


def word_frequencies(text: str, top: int = 5) -> List[Tuple[str, int]]:
    """Return the ``top`` most common words (lowercased; apostrophes kept inside words)."""
    words = re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)*", text.lower())
    return Counter(words).most_common(top)


_ROMAN = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
          (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]


def to_roman(number: int) -> str:
    """Convert an integer in 1..3999 to a Roman numeral."""
    if not 1 <= number <= 3999:
        raise ValueError("number must be between 1 and 3999")
    parts = []
    for value, symbol in _ROMAN:
        count, number = divmod(number, value)
        parts.append(symbol * count)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------

class Graph:
    """A weighted directed graph stored as adjacency lists."""

    def __init__(self) -> None:
        self._edges: Dict[str, List[Tuple[str, float]]] = {}

    def add_edge(self, source: str, target: str, weight: float = 1.0, both_ways: bool = False) -> None:
        """Add an edge; with ``both_ways`` the reverse edge is added too."""
        self._edges.setdefault(source, []).append((target, weight))
        self._edges.setdefault(target, [])
        if both_ways:
            self._edges[target].append((source, weight))

    def nodes(self) -> List[str]:
        return list(self._edges)

    def neighbors(self, node: str) -> List[Tuple[str, float]]:
        return list(self._edges.get(node, []))

    def bfs_path(self, start: str, goal: str) -> Optional[List[str]]:
        """Return a path with the fewest edges from ``start`` to ``goal``, or None."""
        parents: Dict[str, Optional[str]] = {start: None}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node == goal:
                path = []
                while node is not None:
                    path.append(node)
                    node = parents[node]
                return path[::-1]
            for neighbor, _ in self._edges.get(node, []):
                if neighbor not in parents:
                    parents[neighbor] = node
                    queue.append(neighbor)
        return None

    def dijkstra(self, start: str) -> Dict[str, float]:
        """Return shortest distances from ``start`` (weights must be non-negative)."""
        distances = {start: 0.0}
        heap = [(0.0, start)]
        while heap:
            dist, node = heapq.heappop(heap)
            if dist > distances[node]:
                continue  # stale heap entry
            for neighbor, weight in self._edges.get(node, []):
                if weight < 0:
                    raise ValueError("Dijkstra's algorithm needs non-negative weights")
                candidate = dist + weight
                if candidate < distances.get(neighbor, math.inf):
                    distances[neighbor] = candidate
                    heapq.heappush(heap, (candidate, neighbor))
        return distances

    def topological_order(self) -> List[str]:
        """Return the nodes in topological order (Kahn's algorithm); raise on cycles."""
        indegree = {node: 0 for node in self._edges}
        for edges in self._edges.values():
            for target, _ in edges:
                indegree[target] += 1
        ready = deque(node for node, degree in indegree.items() if degree == 0)
        order = []
        while ready:
            node = ready.popleft()
            order.append(node)
            for target, _ in self._edges[node]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
        if len(order) != len(self._edges):
            raise ValueError("graph has a cycle")
        return order


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class LRUCache(Generic[K, V]):
    """A fixed-capacity mapping that evicts the least recently used key."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._data: "OrderedDict[K, V]" = OrderedDict()

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        if key not in self._data:
            return default
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key: K, value: V) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        if len(self._data) > self.capacity:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data


# ---------------------------------------------------------------------------
# Dataclasses: a tiny inventory
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Product:
    """An immutable product description; ``price`` is in the smallest currency unit."""

    sku: str
    name: str
    price: int


@dataclass
class Inventory:
    """Stock levels keyed by SKU."""

    products: Dict[str, Product] = field(default_factory=dict)
    stock: Dict[str, int] = field(default_factory=dict)

    def add(self, product: Product, quantity: int = 0) -> None:
        if quantity < 0:
            raise ValueError("quantity must be non-negative")
        self.products[product.sku] = product
        self.stock[product.sku] = self.stock.get(product.sku, 0) + quantity

    def remove(self, sku: str, quantity: int) -> None:
        available = self.stock.get(sku, 0)
        if quantity > available:
            raise ValueError(f"only {available} units of {sku} in stock")
        self.stock[sku] = available - quantity

    def total_value(self) -> int:
        return sum(self.products[sku].price * count for sku, count in self.stock.items())

    def low_stock(self, threshold: int = 5) -> List[str]:
        return sorted(sku for sku, count in self.stock.items() if count < threshold)


# ---------------------------------------------------------------------------
# Expression evaluation: shunting-yard to reverse Polish notation
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"\s*(?:(\d+\.?\d*|\.\d+)|(.))")
_PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2, "^": 3}


def tokenize(expression: str) -> List[str]:
    """Split an arithmetic expression into number, operator and parenthesis tokens."""
    tokens = []
    for number, other in _TOKEN.findall(expression):
        token = number or other
        if token.strip():
            tokens.append(token)
    return tokens


def to_rpn(tokens: Sequence[str]) -> List[str]:
    """Convert infix tokens to reverse Polish notation (``^`` is right-associative)."""
    output: List[str] = []
    operators: List[str] = []
    for token in tokens:
        if token in _PRECEDENCE:
            while (operators and operators[-1] in _PRECEDENCE and
                   (_PRECEDENCE[operators[-1]] > _PRECEDENCE[token] or
                    (_PRECEDENCE[operators[-1]] == _PRECEDENCE[token] and token != "^"))):
                output.append(operators.pop())
            operators.append(token)
        elif token == "(":
            operators.append(token)
        elif token == ")":
            while operators and operators[-1] != "(":
                output.append(operators.pop())
            if not operators:
                raise ValueError("unbalanced parentheses")
            operators.pop()
        else:
            float(token)  # raises ValueError for unknown symbols
            output.append(token)
    while operators:
        op = operators.pop()
        if op == "(":
            raise ValueError("unbalanced parentheses")
        output.append(op)
    return output


def evaluate(expression: str) -> float:
    """Evaluate an arithmetic expression with + - * / ^ and parentheses."""
    stack: List[float] = []
    for token in to_rpn(tokenize(expression)):
        if token in _PRECEDENCE:
            if len(stack) < 2:
                raise ValueError("malformed expression")
            right, left = stack.pop(), stack.pop()
            if token == "+":
                stack.append(left + right)
            elif token == "-":
                stack.append(left - right)
            elif token == "*":
                stack.append(left * right)
            elif token == "/":
                stack.append(left / right)
            else:
                stack.append(left ** right)
        else:
            stack.append(float(token))
    if len(stack) != 1:
        raise ValueError("malformed expression")
    return stack[0]


# ---------------------------------------------------------------------------
# Self-test and demo
# ---------------------------------------------------------------------------

def _self_test() -> None:
    assert gcd(48, 18) == 6 and lcm(4, 6) == 12 and lcm(0, 5) == 0
    assert [n for n in range(30) if is_prime(n)] == primes_up_to(29)
    assert prime_factors(360) == [2, 2, 2, 3, 3, 5]
    assert fibonacci(10) == 55
    data = [5, 3, 9, 1, 5, 8, 2]
    assert merge_sort(data) == quicksort(data) == sorted(data)
    assert binary_search(sorted(data), 8) == 5 and binary_search(sorted(data), 4) == -1
    assert edit_distance("kitten", "sitting") == 3
    assert knapsack([1, 3, 4, 5], [1, 4, 5, 7], 7) == 9
    assert is_palindrome("A man, a plan, a canal: Panama") and not is_palindrome("Moonlight")
    assert word_frequencies("It's what it is, it's fine", top=1) == [("it's", 2)]
    assert to_roman(1994) == "MCMXCIV" and to_roman(2024) == "MMXXIV"
    graph = Graph()
    for a, b, w in [("A", "B", 4), ("A", "C", 1), ("C", "B", 2), ("B", "D", 1), ("C", "D", 5)]:
        graph.add_edge(a, b, w)
    assert graph.dijkstra("A") == {"A": 0.0, "B": 3.0, "C": 1.0, "D": 4.0}
    assert graph.bfs_path("A", "D") == ["A", "B", "D"]
    assert graph.topological_order()[0] == "A"
    cache: LRUCache[str, int] = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")
    cache.put("c", 3)
    assert "b" not in cache and cache.get("a") == 1
    shop = Inventory()
    shop.add(Product("P1", "Notebook", 12000), 10)
    shop.add(Product("P2", "Pen", 3000), 2)
    shop.remove("P1", 4)
    assert shop.total_value() == 78000 and shop.low_stock() == ["P2"]
    assert evaluate("2 + 3 * (4 - 1) ^ 2") == 29.0
    assert evaluate("2 ^ 3 ^ 2") == 512.0


def main() -> None:
    _self_test()
    print("All self-tests passed.")
    print("Primes below 50:", primes_up_to(49))
    print("Roman numeral for 2024:", to_roman(2024))
    print("Edit distance kitten -> sitting:", edit_distance("kitten", "sitting"))
    print("Top words:", word_frequencies("the cat and the hat and the bat", top=2))
    print("(1 + 2) * 3.5 =", evaluate("(1 + 2) * 3.5"))


if __name__ == "__main__":
    main()
