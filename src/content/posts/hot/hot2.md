---
title: 力扣 hot100 - 双指针
published: 2026-02-25
description: "hot100"
image: "./cover1.png"
tags: ["hot100"]
category: hot100
draft: false
---

## 283. 移动零

开一个指针寻找并定位到当前最左边 $0$ 的位置，另一个指针向后找，遇到第一个非 $0$ 的数字后，两个指针位置的数字进行交换，同时第一个指针继续寻找并定位到当前最左边 $0$ 的位置，以此类推，像冒泡一样把非 $0$ 的数字冒到左边来。

代码：
```cpp
class Solution {
public:
    void moveZeroes(vector<int>& nums) {
        int zero = 0;
        for (int i = 0; i < nums.size(); i++) {
            if (nums[i]) {
                swap(nums[zero], nums[i]);
                zero++;
            }
        }
    }
};
```

## 11. 盛最多水的容器

考虑左右指针一开始分别指向左右两端，逐渐向中间合并，考虑左右指针指向的线的高度，若较高一方向指针另一方移动，无论如何盛水体积都不会比现在更多，所以每次记录容器容积后，选较矮一方的指针向另一方移动，两边互相逼近即可。

代码：
```cpp
class Solution {
public:
    int maxArea(vector<int>& height) {
        int l = 0, r = height.size() - 1, ans = 0;;
        while (l < r) {
            ans = max(ans, (r - l) * min(height[l], height[r]));
            height[l] < height[r] ? l++ : r--;
        }
        return ans;
    }
};
```

---
施工中。。。