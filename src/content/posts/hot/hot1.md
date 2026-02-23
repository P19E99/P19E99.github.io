---
title: 力扣 hot100 - 哈希
published: 2026-02-23
description: "hot100"
image: "./cover1.png"
tags: ["hot100"]
category: hot100
draft: false
---

## 1. 两数之和

### 暴力

看到 $nums$ 的长度只有 $10^4$ 考虑 $O(n^2)$ 的暴力，枚举数组中任意两数加和看等不等于 $target$。

代码：
```cpp
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        int n = nums.size();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] + nums[j] == target) return {i, j};
            }
        }
        return {1, 1, 4, 5, 1, 4};   //理论上一定有解所以出不了循环，但是这里不return，会编译错误
    }
};
```

### 哈希

上面的做法太暴力了，我们想一下有没有什么办法做做优化。

考虑到我们可以把遍历过的数存到一个哈希表中，表里记录的是他的位置，那么我们在遍历到后面的数字 $x$ 的时候，我们只需要看 $target-x$ 在不在表里就好了，如果不在的话就把 $x$ 丢到表里继续往下遍历，这样我们就能把时间复杂度降到 $O(n)$ 。

代码：
```cpp
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        int n = nums.size();
        unordered_map<int, int> mp;
        for (int i = 0; i < n; i++) {
            if (mp.count(target - nums[i])) return {mp[target - nums[i]], i};
            mp[nums[i]] = i;
        }
        return {1, 1, 4, 5, 1, 4};
    }
};
```

## 49. 字母异位词分组

其实做法在第一个样例的解释中已经写的差不多了，给相同字母的字符串分组开哈希，对每个字符串排个序往哈希表里对一下就行，很暴力的解法，时间复杂度上也过得去。

代码：
```cpp
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        unordered_map<string, vector<string>> mp;
        for (auto s : strs) {
            string t = s;
            ranges::sort(s);
            mp[s].push_back(t);
        }
        vector<vector<string>> ans;
        for (auto& [k, v] : mp) ans.push_back(move(v));
        return ans;
    }
};
```

## 128. 最长连续序列

元素范围是 $10^9$ 挨个数去暴力扩展肯定不中，那接下来思考怎么快速找连续段。

首先考虑到，连续段肯定有个头，如果我们抓着这个头去进行扩展会不会好很多，那我们怎么找到这个头呢？

我们只需要把数组塞进哈希，对于元素 $x$，看 $x-1$ 在不在表里就行了。如果不在说明 $x$ 就是一个连续段的头，我们对于这样的头做扩展，看 $x+1$、$x+2$ 等等在不在表里。对于每个连续段记一下长度即可。

代码：
```cpp
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        unordered_set<int> s;
        for (auto i : nums) s.insert(i);
        int ans = 0;
        for (auto st : s) {
            if (s.count(st - 1)) continue;
            int ed = st + 1;
            while (s.count(ed)) ed++;
            ans = max(ans, ed - st);
        }
        return ans;
    }
};
```