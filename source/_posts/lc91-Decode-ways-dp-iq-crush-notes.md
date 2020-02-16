---
title: lc91_Decode_ways-dp-iq_crush_notes
date: 2020-02-15 23:29:45
tags: leetcode DynamicProgrammming

---

Problem: https://leetcode.com/problems/decode-ways/

Solution: https://leetcode.com/problems/decode-ways/discuss/253018/Python%3A-Easy-to-understand-explanation-bottom-up-dynamic-programming

The problem can be abstract into two basic situations: 2-int as 1 number; 2 in as 2 numbers.

Let dp[i] be the possible decode ways to string s[:i - 1]. Make sure len(dp) = len(s) + 1 and dp[0] = 1, in case the tricky corner case like "10".

So if s[i - 1] != "0", dp[i] would be 0 + dp[i - 1], as a single number for decoding.

Then if s[i - 2:i] is a valid code in range of (1, 26), s[i - 1] can be treat as the other way to decode together with s[i - 2]. Thus, dp[i] should add the decode ways of dp[i - 2]. dp[i - 2] and dp[i - 1] are not overlapped.

```python
class Solution:
    def numDecodings(self, s: str) -> int:
        if s[0] == "0":
            return 0
        n = len(s)
        dp = [0 for _ in range(n + 1)]
        dp[0] = 1
        dp[1] = 1
        for i in range(2, n + 1):
            if s[i-2:i] == "00" or (s[i - 1] == "0" and int(s[i-2:i]) > 26):
                return 0
            if s[i - 1] != "0":
                dp[i] += dp[i - 1]
            if 10 <= int(s[i-2:i]) <= 26:
                dp[i] += dp[i - 2]
        return dp[-1]
```



There is another solution come to my mind before, which is using dfs.

```python
class Solution:
    def numDecodings(self, s: str) -> int:
        return self._dfs(s, 0)
    
    def _dfs(self, rest, ans):
        if not rest:
            return ans + 1
        for i in range(1, min(3, len(rest) + 1)):
            if rest[:i][0] == "0":
                continue
            if rest[:i] and int(rest[:i]) > 0 and int(rest[:i]) <= 26:
                ans = self._dfs(rest[i:], ans)
        return ans
```

This dfs solution can perfectly avoid the corner cases by searching all possible 1-2 integers as the solution.

However, the time complexity is 

<img src="http://chart.googleapis.com/chart?cht=tx&chl= O(\sum^{n}_{k = 1}2^k)" style="border:none;">

which is much higher than dp solution O(n).

Space complexity is O(n) both in bfs and dp.