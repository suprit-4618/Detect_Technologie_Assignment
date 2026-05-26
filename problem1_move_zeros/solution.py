"""
Problem 1 — Easy: Move Zeros to the Right
==========================================
Given an array of integers, move all zeros to the end while keeping
the order of non-zero elements intact.

Example:
    [0, 1, 0, 3, 12] → [1, 3, 12, 0, 0]

Approach:
    Two-pointer technique (in-place).
    - `insert_pos` tracks where the next non-zero element should go.
    - Single left-to-right pass; swap non-zero elements to the front.
    - All remaining positions are already zero after the pass.

Time Complexity:  O(n)  — single pass through the array
Space Complexity: O(1)  — in-place, no extra data structure used
"""


def move_zeros_to_right(arr: list[int]) -> list[int]:
    """
    Move all zeros to the end of the array while preserving
    the relative order of non-zero elements.

    Args:
        arr: List of integers (may include negatives and zeros).

    Returns:
        The same list modified in-place with zeros at the end.
    """
    insert_pos = 0  # Points to the slot for the next non-zero element

    # Pass 1: shift all non-zero elements to the front
    for i in range(len(arr)):
        if arr[i] != 0:
            arr[insert_pos] = arr[i]
            insert_pos += 1

    # Pass 2 (continuation): fill the rest with zeros
    for i in range(insert_pos, len(arr)):
        arr[i] = 0

    return arr


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def run_tests():
    test_cases = [
        # (input,                    expected_output,          description)
        ([0, 1, 0, 3, 12],          [1, 3, 12, 0, 0],         "Basic example"),
        ([0, 0, 0, 0],              [0, 0, 0, 0],              "All zeros"),
        ([1, 2, 3, 4],              [1, 2, 3, 4],              "No zeros"),
        ([-1, 0, -3, 0, 5],        [-1, -3, 5, 0, 0],        "With negative numbers"),
        ([0],                        [0],                       "Single zero"),
        ([1],                        [1],                       "Single non-zero"),
        ([],                         [],                        "Empty array"),
        ([0, 0, 1],                 [1, 0, 0],                 "Zeros at start"),
        ([1, 0, 0],                 [1, 0, 0],                 "Zeros at end"),
        ([1, 0, 2, 0, 3, 0, 4, 0], [1, 2, 3, 4, 0, 0, 0, 0], "Alternating zeros"),
    ]

    print("=" * 60)
    print("Running Test Cases for Move Zeros to the Right")
    print("=" * 60)

    all_passed = True
    for i, (inp, expected, desc) in enumerate(test_cases, 1):
        result = move_zeros_to_right(inp.copy())
        status = "PASS" if result == expected else "FAIL"
        if result != expected:
            all_passed = False
        print(f"Test {i:2d} [{status:<4}] {desc}")
        if result != expected:
            print(f"         Input:    {inp}")
            print(f"         Expected: {expected}")
            print(f"         Got:      {result}")

    print("-" * 60)
    print("All tests PASSED" if all_passed else "Some tests FAILED")
    print("=" * 60)

    # Complexity summary
    print("\nComplexity Analysis:")
    print("  Time Complexity  : O(n) — two linear passes over the array")
    print("  Space Complexity : O(1) — in-place modification, no extra memory")


if __name__ == "__main__":
    # Interactive demo
    print("\nDemo:")
    sample = [0, 1, 0, 3, 12]
    print(f"  Input:  {sample}")
    result = move_zeros_to_right(sample)
    print(f"  Output: {result}")
    print()

    run_tests()
