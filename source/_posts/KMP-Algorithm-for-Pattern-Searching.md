---
title: KMP_Algorithm_for_Pattern_Searching
date: 2020-05-01 17:53:28
tags: Algorithms
---

### Problem

Given two strings s and p, find the first position that p can be totally matched in s. 

For example, s = "BBC ABCDAB ABCDABCDABDE", p = "ABCDABD"

The first matching position is supposed to be s[15]. Thus, it should return a 15 as result.

### Solution

Normally, we will think about the brutal force, which will take O(mn) time complexity.

However, KMP algorithm is a O(m+n) time complexity algorithm for pattern matching. 

The code comes from [this blog](https://blog.csdn.net/v_JULY_v/article/details/7041827).

```python
def KMP_match(s, p):
    if len(s) < len(p):
        s, p = p, s
    m, n = len(s), len(p)
    next = get_next(p)
    i = j = 0
    while i < m and j < n:
        if s[i] == p[j] or j == -1:
            i += 1
            j += 1
        else:
            j = next[j]
    if j == n:
        return i - j
    return -1
  
def get_next(p):
    n = len(p)
    next = [0] * len(p)
    next[0] = -1
    k = -1
    i = 0
    while i < n - 1:
        if k == -1 or p[i] == p[k]:
            i += 1
            k += 1
            next[i] = k
        else:
            k = next[k]
    return next
```

Basiclly, the algorithm memorized the scanned sub-patterns in p. So, when a mismatching appeared, the pointer j on p can move to the closest shown subpattern in p. The worst case j will move backward for n times, where n is the length of p. After that, the pointer i on s will move forward until find the next match point.

The hardest part is to get the next array from p. It is a list to save the length of matchd prefix and suffix to the substring of array. Also look at [this blog](http://jakeboxer.com/blog/2009/12/13/the-knuth-morris-pratt-algorithm-in-my-own-words/). Just remember it if can't understand it. 

A faster version by changing get_next function

```python
def get_next(p):
    n = len(p)
    next = [0] * len(p)
    next[0] = -1
    k = -1
    i = 0
    while i < n - 1:
        if k == -1 or p[i] == p[k]:
            i += 1
            k += 1
            if p[i] != p[k]:
                next[i] = k
            else:
                next[i] = next[k]
        else:
            k = next[k]
    return next
```



 