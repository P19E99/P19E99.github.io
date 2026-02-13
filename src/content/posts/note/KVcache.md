---
title: 【八股】KV Cache
published: 2026-02-13
description: "KV Cache"
image: "./cover5.jpg"
tags: ["Interview Notes"]
category: Interview Notes
draft: false
---

## 1. 啥是 KV cache

KV cache 是 Transformer 在自回归生成（decoder-only 或 encoder-decoder 的 decoder 侧）做推理加速的核心机制：把历史 token 在每一层注意力里算出来的 Key/Value 张量缓存起来，后续每生成一个新 token 时直接复用这些 K/V，而不是把整段上下文再前向一遍去重算一遍，是一种空间换时间的优化。

## 2. KV cache 是怎么来的

既然 KV cache 是 decoder 推理时的优化，我们不妨看看优化前的推理过程有哪些不足的地方。

自回归解码（decoder-only 或 decoder 部分）在第 $t$ 步：给定前缀 $x_{1:t-1}$，需要计算当前位置的表示并预测 $x_t$。

在某一层 self-attention（先忽略多头，或把它理解成对每个 head 独立做同样的事），对该层输入隐状态矩阵
$$
H_{1:t} \in \mathbb{R}^{t \times d_{\text{model}}}
$$
做三次线性投影：
$$
Q_{1:t}=H_{1:t}W_Q,\quad K_{1:t}=H_{1:t}W_K,\quad V_{1:t}=H_{1:t}W_V.
$$

注意力输出为：
$$
A_{1:t}=\mathrm{softmax}\left(\frac{Q_{1:t}K_{1:t}^{\top}}{\sqrt{d_k}} + M\right),\qquad
O_{1:t}=A_{1:t}V_{1:t},
$$
其中 $M$ 是 causal mask（上三角为 $-\infty$），保证只能看历史。

不加 KV cache 的问题在于当从 $t-1$ 生成到 $t$ 时，历史部分 $H_{1:t-1}$ 在同一条生成轨迹里并不会变化，但你仍会在每一层、每一步重复计算它们对应的 $K_{1:t-1}$ 和 $V_{1:t-1}$（以及相关张量拼接），这就是大量冗余。

KV cache 的想法是只缓存未来还会反复被用到的量，也就是历史 token 的 $K,V$。

把第 $t$ 步拆开写：

1. 只对新 token 的隐状态 $h_t\in\mathbb{R}^{d_{\text{model}}}$ 投影：
$$
q_t=h_tW_Q,\quad k_t=h_tW_K,\quad v_t=h_tW_V.
$$

2. 维护缓存（逐步 append）：
$$
K_{\text{cache}}^{(t)}=\mathrm{concat}\left(K_{\text{cache}}^{(t-1)},,k_t\right),\quad
V_{\text{cache}}^{(t)}=\mathrm{concat}\left(V_{\text{cache}}^{(t-1)},,v_t\right).
$$

3. 第 $t$ 步输出（只需要当前 query 对全历史 keys 做匹配）：
$$
\alpha_t=\mathrm{softmax}\left(\frac{q_t\left(K_{\text{cache}}^{(t)}\right)^{\top}}{\sqrt{d_k}}\right),\qquad
o_t=\alpha_t,V_{\text{cache}}^{(t)}.
$$

那为什么只对 $K,\ V$ 进行缓存而不缓存 $Q$ ？ 

因为第 $t$ 步只用到当前 $q_t$ 去查历史 $K$，过去的 $q_{1:t-1}$ 不会在后续步骤再被用到，因此缓存它没有复用价值。

## 3. 空间占用分析

>这部分参考了 satsuki26681534 的 [【大模型】什么是KV cache](https://www.cnblogs.com/satsuki26681534/p/19449217)

可以这样算 KV cache 所占用的空间：

$$
每个\ token\ 的缓存=2\times 层数\times 隐藏维度\times 数据类型大小
$$

举个栗子：

```text
层数 L = 32
隐藏维度 d = 4096
头数 h = 32
每个头维度 d_h = 128
数据类型: float16 (2字节)

每个token每层缓存大小 = 2 × (h × d_h) × 2字节
                         = 2 × 4096 × 2
                         = 16,384字节 ≈ 16KB

每个token总缓存 = 16KB × 32层 = 512KB

生成1000个token的缓存 = 512KB × 1000 = 512MB
```

## 4. 代码实现（带 KV cache 的 MHA）

下面代码在之前笔记中的 MHA 代码基础上修改。

```python
class MultiHeadAttention(nn.Module):
	def __init__(self, input_dim, num_heads, dim_qk, dim_v):
		super().__init__()
		self.num_heads = num_heads
		self.head_dim_qk = dim_qk // num_heads
		self.head_dim_v = dim_v // num_heads
		
		self.q = nn.Linear(input_dim, dim_qk)
		self.k = nn.Linear(input_dim, dim_qk)
		self.v = nn.Linear(input_dim, dim_v)
		
		self.scale = sqrt(self.head_dim_qk)
		
		self.out = nn.Linear(dim_v, input_dim)

        #新增，初始化 cache
        self.cache_k = None
        self.cache_v = None
		
	def forward(self, x, use_cache = False):
		batch, seq = x.shape[:2]
		
		q = self.q(x)
		k = self.k(x)
		v = self.v(x)
		
		q = q.view(batch, seq, self.num_heads, self.head_dim_qk).transpose(1, 2)
		k = k.view(batch, seq, self.num_heads, self.head_dim_qk).transpose(1, 2)
		v = v.view(batch, seq, self.num_heads, self.head_dim_v).transpose(1, 2)

        #新增，如果使用缓存且缓存非空则拼接 KV
        if use_cache and self.cache_k is not None:
            k = torch.cat([self.cache_k, k], dim = -2)
            v = torch.cat([self.cache_v, v], dim = -2)
        #新增，更新 cache
        if use_cache:
            self.cache_k = k
            self.cache_v = v
		
		scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
		weights = torch.softmax(scores, dim = -1)
		
		out = torch.matmul(weights, v)
		out = out.transpose(1, 2).contiguous().view(batch, seq, -1)
		out = self.out(out)
		
		return out
```