---
title: 力扣 hot100 - 双指针
published: 2026-03-12
description: "hot100"
image: "./cover2.png"
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

## 15. 三数之和

看数据范围可以在 $O(n^2)$ 的复杂度内解决问题。

所以我们先花 $O(n)$ 枚举数组内的每一个数 $nums_i$ ，这样就可以把问题简化成在数组的其他数里找出两个跟他的和是 $0$ 的两个数。那怎么用剩下的 $O(n)$ 完成找两个数的操作呢。

考虑排序和双指针，我们将数组排序后取两个指针 $l$ 和 $r$ 分别指向 $nums_i$ 的下一个数字和数组最后一个数字，这时候我们可以发现：

$$
nums_i+nums_l+nums_r\leq nums_i+nums_{l+1}+nums_r
$$

和

$$
nums_i+nums_l+nums_r\leq nums_i+nums_l+nums_{r-1}
$$

这样我们令 $sum=nums_i+nums_l+nums_r$ ，如果 $sum < 0$ ，则让 $l+1$ ，如果 $sum > 0$ ，则让 $r-1$ 。直到 $sum=0$ ，这时我们跳过 $l$ 后面和他相同的所有数和 $r$ 前面和他相同的所有数，保证不重复即可。

代码：
```cpp
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        int n = nums.size();
        ranges::sort(nums);
        vector<vector<int>> ans;
        for (int i = 0; i < n; i++) {
            if (i != 0 && nums[i] == nums[i - 1]) continue;
            int l = i + 1, r = n - 1;
            while (l < r) {
                int sum = nums[i] + nums[l] + nums[r];
                if (sum < 0) l++;
                else if (sum > 0) r--;
                else {
                    ans.push_back({nums[i], nums[l], nums[r]});
                    l++, r--;
                    while (l < r && nums[l - 1] == nums[l]) l++;
                    while (l < r && nums[r + 1] == nums[r]) r--;
                }
            }
        }
        return ans;
    }
};
```

## 42. 接雨水

每个位置能存多少水取决于该位置左侧和右侧最高的柱子中较矮的那个，我们令位置 $i$ 左侧最高的柱子高度为 $l\_max_i$ ，右侧最高的柱子高度为 $r\_max_i$ ，那么位置 $i$ 能存的水的量为：

$$
\text{min}(l\_max_i,r\_max_i)-height_i
$$

朴素做法是,对每个位置都往左、往右找最高柱子，再计算答案，但这样时间复杂度是 $O(n^2)$。
更好的做法是用双指针 $l$ 和 $r$ 从两侧往中间夹，同时维护左右两侧最高的柱子 $lm$ 和 $rm$ 。

如果 $lm<rm$ 则此时 $l$ 位置的存水量就可以决定了，反之则 $r$ 位置的存水量可以确定，累加到答案里之后向中间逼近即可。

```cpp
class Solution {
public:
    int trap(vector<int>& height) {
        int l = 0, r = height.size() - 1, lm = 0, rm = 0, ans = 0;
        while (l < r) {
            lm = max(lm, height[l]);
            rm = max(rm, height[r]);
            ans += lm < rm ? lm - height[l++] : rm - height[r--];
        }
        return ans;
    }
};
```