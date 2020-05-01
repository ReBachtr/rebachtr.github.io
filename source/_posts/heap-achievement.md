---
title: heap_achievement
date: 2020-04-30 17:15:12
tags: Algorithms
---

Heap is a common data structure as a kind of binary tree. Here we are aiming at the minimum heap, which follows the rules that the value of root is less than any its node in the tree, as well as the roots of any subtrees in the heap.

Achievement of the heap using python is only relying on the data structure list.

By the index $i$ of any element in the list, we can get its parent node index as 

<img src="http://chart.googleapis.com/chart?cht=tx&chl= i_{parent} = (i - 1) // 2" style="border:none;">

$i_{parent} = (i - 1) // 2$ and its child nodes 

<img src="http://chart.googleapis.com/chart?cht=tx&chl= $i_{left}=2*i + 1$" style="border:none;">

$i_{left} = 2 * i + 1$, $i_{right} = 2 * i + 2$

<img src="http://chart.googleapis.com/chart?cht=tx&chl= i_{right} = 2 * i + 2" style="border:none;">

There are two adjustment strategies for keeping the rules of this minimum heap.

One is switching the position from up to bottom

```python
def ub_adjust(nums, i):
    n = len(nums)
    left = 2 * i + 1
    right = 2 * i + 2
    min_index = i
    if left < n and nums[min_index] > nums[left]:
        min_index = left
    if right < n and nums[min_index] > nums[right]:
        min_index = right
    if min_index != i:
        nums[min_index], nums[i] = nums[i], nums[min_index]
        ub_adjust(nums, min_index)
    return 
```

Adjust bottom-up

```python
def bu_adjust(nums, i):
    if i == 0:
        return 
    p = (i - 1)//2
    if nums[p] > nums[i]:
        nums[i], nums[p] = nums[p], nums[i]
        bu_adjust(nums, p)
    return
```

Thus, given a array list nums, it can be heapified by using up-bottom adjustment level by level

``` python
def heapify(nums):
    n = len(nums)
    for i in range(int(n/2), -1, -1):
        ub_adjust(nums, i)
    return nums
```

Inserting element is adding an element to the tail of the list and then use bottom-up strategy

```python
def heappush(nums, n):
    nums.append(n)
    i = len(nums) - 1
    bu_adjust(nums, i)
    return
```

Pop out the smallest element use the up-bottom strategy just like heapify

```python
def heappop(nums):
    res = nums[0]
    nums[0] = nums.pop()
    ub_adjust(nums, 0)
    return res   
```

The time complexity of any single update (both bottom-up and up-bottom) is O(logn), since the total number of levels to the heap is logn. However, the time complexity of  building a heap (heapify) is O(n), from [wikipedia](https://en.wikipedia.org/wiki/Binary_heap). And the insertion and pop out are taking O(logn). Looking for the minimum element is only O(1) by simply calling

```python
nums[0]
```



