"""
Problem 1 — Easy: Move Zeros to the Right
==========================================
Given an array of integers, move all zeros to the end while keeping
the order of non-zero elements intact.

Example:
    [-1, -2, -1, -2] → [-1, -2, -1, -2]   (no zeros, unchanged)
    [0, 1, 0, 3, 12]  → [1, 3, 12, 0, 0]

Approach:
    1. Collect all non-zero elements into a new list.
    2. Count how many zeros were removed.
    3. Append that many zeros at the end.

Time Complexity:  O(n)  — single pass to collect non-zeros + one pass to append zeros
Space Complexity: O(n)  — extra list to hold the result
"""


def move_zeros_to_right(a: list) -> list:
    d = []

    for i in a:
        if i != 0:
            d.append(i)

    zeros = len(a) - len(d)

    for i in range(zeros):
        d.append(0)

    return d


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def run_tests():
    test_cases = [
        ([0, 1, 0, 3, 12],          [1, 3, 12, 0, 0],         "Basic example"),
        ([-1, -2, -1, -2],          [-1, -2, -1, -2],         "No zeros, all negative"),
        ([0, 0, 0, 0],              [0, 0, 0, 0],              "All zeros"),
        ([1, 2, 3, 4],              [1, 2, 3, 4],              "No zeros"),
        ([-1, 0, -3, 0, 5],        [-1, -3, 5, 0, 0],        "With negative numbers"),
        ([0],                        [0],                       "Single zero"),
        ([1],                        [1],                       "Single non-zero"),
        ([],                         [],                        "Empty array"),
        ([0, 0, 1],                 [1, 0, 0],                 "Zeros at start"),
        ([1, 0, 2, 0, 3, 0, 4, 0], [1, 2, 3, 4, 0, 0, 0, 0], "Alternating zeros"),
    ]

    print("=" * 55)
    print("Running Test Cases — Move Zeros to the Right")
    print("=" * 55)

    all_passed = True
    for i, (inp, expected, desc) in enumerate(test_cases, 1):
        result = move_zeros_to_right(inp)
        status = "PASS" if result == expected else "FAIL"
        if result != expected:
            all_passed = False
        print(f"Test {i:2d} [{status}] {desc}")
        if result != expected:
            print(f"         Input:    {inp}")
            print(f"         Expected: {expected}")
            print(f"         Got:      {result}")

    print("-" * 55)
    print("All tests PASSED" if all_passed else "Some tests FAILED")
    print("=" * 55)
    print()
    print("Complexity:")
    print("  Time  : O(n)")
    print("  Space : O(n)")


if __name__ == "__main__":
    # Demo
    sample = [0, 1, 0, 3, 12]
    print(f"Input:  {sample}")
    print(f"Output: {move_zeros_to_right(sample)}")
    print()
    run_tests()
