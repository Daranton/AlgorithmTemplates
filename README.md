# AlgorithmTemplates.py

This file provides a comprehensive set of Python templates for common algorithms and data structures. Each function is designed as a starting point for solving typical coding interview and competitive programming problems. The templates are well-commented and organized by technique, making it easy to find and adapt the right approach for your needs.

## Included Templates

- **Two-pointer techniques:** For single or dual array problems.
- **Sliding window:** For contiguous subarray problems.
- **Prefix sum:** For cumulative sum calculations.
- **Binary search:** Standard, insert position, and range queries.
- **Efficient string building:** For fast string concatenation.
- **Linked list operations:** Fast/slow pointer and reversal.
- **Subarray counting:** Using prefix sums and hashmaps.
- **Monotonic stack:** For problems like Next Greater Element.
- **Tree traversal:** Iterative DFS and BFS.
- **Graph traversal:** Recursive DFS for graphs.
- **Trie:** Node class and builder function.
- **Graph algorithms:** Dijkstra's shortest path and Prim's MST.
- **Dynamic programming:** 1D, 2D, and 3D DP templates.

## Usage

Each function contains TODOs and example logic. Replace these with your problem-specific code.  
Import or copy the relevant template into your solution file and customize as needed.

## Example

```python
from AlgorithmTemplates import binarySearch

arr = [1, 3, 5, 7, 9]
idx = binarySearch(arr, 5)
print(idx)  # Output: 2
```

## Future Plans

In the future, this single file will be separated into multiple files for better modularity and maintainability. Each algorithmic category will have its own dedicated module.

## Contribution

Feel free to submit improvements or additional templates via pull request
