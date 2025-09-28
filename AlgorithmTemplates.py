# --- Two-pointer Templates ---
# Use for problems involving two indices scanning an array or two arrays.

def twoPointerOneInput(arr):
    """
    Template for two-pointer technique on a single array.
    - arr: input array
    - left, right: pointers
    - ans: result variable
    - condition: replace with your problem's condition
    """
    left = ans = 0
    right = len(arr) - 1

    while left < right:
        if condition:  # TODO: define your condition
            left += 1
        else:
            right -= 1

    return ans

def twoPointerTwoInput(arr1, arr2):
    """
    Template for two-pointer technique on two arrays.
    - arr1, arr2: input arrays
    - i, j: pointers
    - ans: result variable
    - condition: replace with your problem's condition
    """
    i = j = ans = 0

    while i < len(arr1) and j < len(arr2):
        if condition:  # TODO: define your condition
            i += 1
        else:
            j += 1
    
    while i < len(arr1):
        # TODO: add logic for remaining arr1
        i += 1

    while j < len(arr2):
        # TODO: add logic for remaining arr2
        j += 1

    return ans

# --- Sliding Window Template ---
# Use for problems involving contiguous subarrays.

def slidingWindow(arr, k):
    """
    Template for sliding window technique.
    - arr: input array
    - k: window size or condition
    - left, right: window boundaries
    - ans: result variable
    - window_condition_broken: replace with your problem's condition
    """
    left = right = ans = 0

    for right in range(len(arr)):
        # TODO: add arr[right] to window

        while window_condition_broken:  # TODO: define your condition
            # TODO: remove arr[left] from window
            left += 1
        # TODO: update ans
    return ans

# --- Prefix Sum Template ---
# Use for problems requiring cumulative sums.

def prefixSum(arr):
    """
    Builds prefix sum array.
    - arr: input array
    - prefix[i]: sum of arr[0] to arr[i-1]
    """
    prefix = [0] * (len(arr) + 1)
    for i in range(len(arr)):
        prefix[i + 1] = prefix[i] + arr[i]
    return prefix

# --- Binary Search Templates ---
# Use for searching in sorted arrays.

def binarySearch(arr, target):
    """
    Standard binary search for target in sorted array.
    Returns index or -1 if not found.
    """
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def binarySearchInsertPosition(arr, target):
    """
    Finds insert position for target in sorted array.
    Returns index where target should be inserted.
    """
    left, right = 0, len(arr)
    while left < right:
        mid = left + (right - left) // 2
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left

def binarySearchRange(arr, target):
    """
    Finds the range [leftIndex, rightIndex] of target in sorted array.
    Returns [-1, -1] if not found.
    """
    def findLeft():
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return left

    def findRight():
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if arr[mid] <= target:
                left = mid + 1
            else:
                right = mid - 1
        return right

    leftIndex = findLeft()
    rightIndex = findRight()
    if leftIndex <= rightIndex:
        return [leftIndex, rightIndex]
    return [-1, -1]

# --- String Building Template ---
# Use for efficient string concatenation.

def EffStringBuild(arr):
    """
    Efficiently builds a string from a list of characters.
    """
    ans = []
    for char in arr:
        ans.append(char)
    return ''.join(ans)

# --- Linked List Templates ---
# Use for linked list problems.

def linkFSpointer(head):
    """
    Fast and slow pointer template (e.g., cycle detection).
    - head: head of linked list
    """
    slow = fast = head
    ans = 0

    while fast and fast.next:
        # TODO: add logic
        slow = slow.next
        fast = fast.next.next
    return slow

def reverseLinkedList(head):
    """
    Reverses a singly linked list.
    - head: head of linked list
    Returns new head.
    """
    prev = None
    curr = head

    while curr:
        next_temp = curr.next
        curr.next = prev
        prev = curr
        curr = next_temp

    return prev

# --- Subarray Counting Template ---
# Use for counting subarrays with a property.

def nSubarrays(arr, k):
    """
    Counts subarrays with a given property using prefix sums and hashmap.
    - arr: input array
    - k: target property
    """
    from collections import defaultdict
    counts = defaultdict(int)
    counts[0] = 1
    ans = curr = 0

    for num in arr:
        # TODO: update curr according to problem
        ans += counts[curr - k]
        counts[curr] += 1
    
    return ans

# --- Monotonic Stack Template ---
# Use for problems like Next Greater Element.

def monotonicStack(arr):
    """
    Monotonic stack template.
    - arr: input array
    - stack: stores indices or values
    """
    stack = []
    ans = 0

    for num in range(len(arr)):
        # for monotonic decreasing stack, just flip > to <
        while stack and stack[-1] > num:
            # TODO: add logic
            stack.pop()
        stack.append(num)

    return ans

# --- Tree Traversal Templates ---
# DFS and BFS for trees.

def DFS(root): # iterative
    """
    Iterative DFS template for tree traversal.
    - root: tree root node
    """
    if not root:
        return

    # TODO: add logic
    DFS(root.left)
    DFS(root.right)
    return ans

from collections import deque

def BFS(root):
    """
    BFS template for tree traversal.
    - root: tree root node
    """
    queue = deque([root])
    ans = 0

    while queue:
        current_length = len(queue)
        # TODO: add logic for each level

        for _ in range(current_length):
            node = queue.popleft()
            # TODO: add logic
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    return ans

# --- Graph Traversal Templates ---
# DFS for graphs.

def DFSrec(graph): # recursive
    """
    Recursive DFS template for graph traversal.
    - graph: adjacency list
    """
    def DFS(node):
        ans = 0
        # TODO: add logic
        for neightbor in graph[node]:
            if neightbor not in seen:
                seen.add(neightbor)
                ans += DFS(neightbor)
        return ans
    seen = {start_node}  # TODO: define start_node
    return DFS(start_node)

# --- Trie Templates ---
# Trie node and builder.

class TrieNode:
    """
    Trie node class for building a trie.
    """
    def __init__(self):
        self.children = {}
        self.data = None

def fn(words):
    """
    Builds a trie from a list of words.
    - words: list of strings
    """
    root = TrieNode()
    for word in words:
        node = root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.data = word
    return root

# --- Graph Algorithms ---
# Dijkstra and Prim's algorithms.

def dijkstra(graph, source):
    """
    Dijkstra's algorithm for shortest path in a graph.
    - graph: adjacency list {node: [(neighbor, weight), ...]}
    - source: starting node
    Returns distances dict.
    """
    from math import inf
    from heapq import heappop, heappush

    distances = {node: inf for node in graph}
    distances[source] = 0
    heap = [(0, source)]

    while heap:
        curr_dist, node = heappop(heap)
        if curr_dist > distances[node]:
            continue

        for neighbor, weight in graph[node]:
            distance = curr_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heappush(heap, (distance, neighbor))
    return distances

def prim_mst(graph, start_node):
    """
    Prim's algorithm for Minimum Spanning Tree (MST).
    - graph: adjacency list {node: [(neighbor, weight), ...]}
    - start_node: starting node
    Returns MST cost.
    """
    from heapq import heappop, heappush
    mst_cost = 0
    visited = set()
    min_heap = [(0, start_node)]  # (cost, node)
    
    while min_heap:
        curr_cost, node = heappop(min_heap)
        if node in visited:
            continue
        visited.add(node)
        mst_cost += curr_cost
    
        for neighbor, weight in graph[node]:
            if neighbor not in visited:
                heappush(min_heap, (weight, neighbor))
    
    return mst_cost

# --- Dynamic Programming Templates ---
# These functions are templates for solving DP problems.
# Replace 'base_case_value' and the example logic with the logic for your specific problem.

# 1D DP template (e.g., House Robber, Fibonacci, etc.)
def dp1D(arr):
    """
    1D Dynamic Programming template.
    - arr: input array (e.g., values, costs, etc.)
    - dp[i]: optimal value for first i elements.
    - base_case_value: set according to your problem (e.g., 0 for sum problems).
    - Example recurrence: dp[i] = max(dp[i-1], dp[i-2] + arr[i-1])
    """
    n = len(arr)
    dp = [0] * (n + 1)
    dp[0] = base_case_value  # TODO: Set base case value for your problem

    for i in range(1, n + 1):
        # TODO: Change recurrence according to your problem
        dp[i] = max(dp[i - 1], dp[i - 2] + arr[i - 1])  # Example: House Robber

    return dp[n]
# 2D DP template (e.g., Grid problems, Edit Distance, etc.)
def dp2D(matrix):
    """
    2D Dynamic Programming template.
    - matrix: 2D input (e.g., grid, cost matrix)
    - dp[r][c]: optimal value for submatrix (r, c)
    - base_case_value: set according to your problem
    - Example recurrence: dp[r][c] = max(dp[r-1][c], dp[r][c-1]) + matrix[r-1][c-1]
    """
    if not matrix or not matrix[0]:
        return 0

    rows, cols = len(matrix), len(matrix[0])
    dp = [[0] * (cols + 1) for _ in range(rows + 1)]
    dp[0][0] = base_case_value  # TODO: Set base case value for your problem

    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            # TODO: Change recurrence according to your problem
            dp[r][c] = max(dp[r - 1][c], dp[r][c - 1]) + matrix[r - 1][c - 1]  # Example: max path sum

    return dp[rows][cols]
# 3D DP template (e.g., 3D grid problems)
def dp3D(arr):
    """
    3D Dynamic Programming template.
    - arr: 3D input array
    - dp[i][j][k]: optimal value for subarray (i, j, k)
    - base_case_value: set according to your problem
    - Example recurrence: dp[i][j][k] = max(dp[i-1][j][k], dp[i][j-1][k], dp[i][j][k-1]) + arr[i-1][j-1][k-1]
    """
    d1, d2, d3 = len(arr), len(arr[0]), len(arr[0][0])
    dp = [[[0] * (d3 + 1) for _ in range(d2 + 1)] for _ in range(d1 + 1)]
    dp[0][0][0] = base_case_value  # TODO: Set base case value for your problem

    for i in range(1, d1 + 1):
        for j in range(1, d2 + 1):
            for k in range(1, d3 + 1):
                # TODO: Change recurrence according to your problem
                dp[i][j][k] = max(
                    dp[i - 1][j][k],
                    dp[i][j - 1][k],
                    dp[i][j][k - 1]
                ) + arr[i - 1][j - 1][k - 1]  # Example: max path sum in 3D

    return dp[d1][d2][d3]

# --- End of Templates ---