---
title: quick_sort_three_ways
date: 2020-02-16 00:15:58
Tags: Algorithms
---

Inspired by [this blog](https://blog.csdn.net/razor87/article/details/71155518), I achieved several ways to build the quick_sort function.
Basic way:
My version use right as pivot
```python
def quick_sort(array, left, right):
    if left >= right:
        return
    head = left
    end = right
    key = array[end]
    while left < right:
        while left < right and array[left] < key:
            left += 1
        array[right] = array[left]
        while left < right and array[right] >= key:
            right -= 1
        array[left] = array[right]
    array[left] = key
    print(array, key)
    quick_sort(array, head, left - 1)
    quick_sort(array, right + 1, end)
```

The version in the blog which use left as pivot.
``` Python
def quick_sort(array, left, right):
    if left >= right:
        return
    low = left
    high = right
    key = array[low]
    while left < right:
        while left < right and array[right] > key:
            right -= 1
        array[left] = array[right]
        while left < right and array[left] <= key:
            left += 1
        array[right] = array[left]
    array[right] = key
    quick_sort(array, low, left - 1)
    quick_sort(array, left + 1, high)
```
```python
Time O(logn)
Space O(1)
```

Another way only uses one single loop:
Use right as pivot
```python
def quick_sort(array, left, right):
    if left < right:
        flag = partition(array, left, right)
        quick_sort(array, left, flag - 1)
        quick_sort(array, flag + 1, right)
        
def partition(array, left, right):
    pivot = array[right]
    i = left - 1
    for j in range(left, right):
        if array[j] <= pivot:
            i += 1
            array[i], array[j] = array[j], array[i]
    array[i + 1], array[right] = array[right], array[i + 1]   
    return i + 1
```

Use left as pivot
```Python
def quick_sort(array, left, right):
    if left < right:
        flag = partition(array, left, right)
        quick_sort(array, left, flag - 1)
        quick_sort(array, flag + 1, right)
        
def partition(array, left, right):
    pivot = array[left]
    i = left
    for j in range(left + 1, right + 1):
        if array[j] < pivot:
            i += 1
            array[i], array[j] = array[j], array[i]
    array[i], array[left] = array[left], array[i]  
    return i
```
```python
Time O(logn)
Space O(1)
```

What's different?
Use the 'for' loop to substitute the 'while' loop.
Put one pointer j into the loop, another i to record the first number greater than key. Exchange the position when j met the requirement, which makes sure the sort process to the key value.

********Update*********
```Python
def quick_sort(array):
    if len(array) < 2:
        return array
    else:
        pivot = array[0]
        smaller = [i for i in array[1:] if i < pivot]
        greater = [j for j in array[1:] if j >= pivot]
    return quick_sort(smaller) + [pivot] + quick_sort(greater)    
```
This method seems much simpler. Howvere, it enhances the usage of memory. The time complexity is the same, but the space is O(logn)