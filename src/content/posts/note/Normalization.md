---
title: 【八股】Normalization
published: 2026-02-12
description: "常见 Normalization 策略，以及 LayerNorm、RMSNorm 手撕"
image: "./cover4.jpg"
tags: ["Interview Notes"]
category: Interview Notes
draft: false
---
## 1. 为什么需要Normalization

Normalization 主要是为了解决“训练不稳定”和“信号/梯度尺度失控”这两类问题。

第一，控制激活尺度，避免越堆越炸。Transformer 一层里有残差连接：输出是 $x + Sublayer(x)$ 。如果每层的 Sublayer 输出尺度有漂移（变大或变小），堆很多层后，hidden states 的方差可能不断累积或逐渐塌缩，导致数值不稳定、注意力 logits 过大/过小、后续层的输入分布越来越偏。LayerNorm 把每个 token 的特征向量拉回到“均值≈0、方差≈1”的相对稳定范围，让每层看到的输入分布更一致。

第二，让梯度更好传，减少梯度爆炸/消失。训练深网络本质上是反向传播乘一串 Jacobian。中间激活分布如果漂移严重，会让某些方向梯度特别大或特别小。通过规范化特征分布，等价于把优化地形“变得更可控”，通常能显著改善收敛速度和稳定性。

## 2. Pre-Norm vs Post-Norm

Pre Norm 和 Post Norm 分别指把 Normalization 操作放到残差连接**之前**和残差连接**之后**。
$$
\begin{aligned}
\text{Pre}\ \text{Norm}: x_{t+1} = x_t+F_t(\text{Norm}(x_t)) \\
\text{Post}\ \text{Norm}: x_{t+1} = \text{Norm}(x_t+F_t(x_t))
\end{aligned}
$$

先说结论，Pre Norm 结构往往更容易训练，但最终效果通常不如 Post Norm。

直觉上，残差网络之所以好训，是因为存在一条接近恒等映射的“高速通道”：如果输出里能直接包含 $x$，梯度就能比较直接地穿透很多层回传。Pre Norm 里，这条通道非常干净：$y = x + \text{(something)}$ ，对 $x$ 的导数里天然有一个“+1”的恒等项，所以深层时梯度不容易断。

Post Norm 把 Normalization 放在残差相加之后，变成 $y=\text{Norm}(x+F(x))$ 。这会让“高速通道”不再是纯粹的恒等映射，因为你无论走残差还是走子层，最后都要再过一次 LN 的 Jacobian（归一化操作对各维耦合）。于是梯度在层间传播时更容易出现幅度失衡：有的层梯度过大导致不稳定，有的层梯度过小导致训练慢或难以优化。


> 下面推导过程参考自苏剑林 [浅谈Transformer的初始化、参数化与标准化](https://kexue.fm/archives/8620) 
> 
> 假设初始状态下 $x,\ F(x)$ 的方差均为 $1$ ，那么  $x+F(x)$  的方差就是 $2$ ，Normalization 负责将方差重新降为 $1$ ，说明初始状态下 Post Norm 相当于：
> 
> $$x_{t+1} = \frac{x_t + F_t(x_t)}{\sqrt{2}}$$
> 
> 递归得到
> 
> $$\begin{aligned}x_l &= \frac{x_{l-1}}{\sqrt{2}} + \frac{F_{l-1}(x_{l-1})}{\sqrt{2}} \\ &= \frac{x_{l-2}}{2} + \frac{F_{l-2}(x_{l-2})}{2} + \frac{F_{l-1}(x_{l-1})}{\sqrt{2}} \\ &= \cdots \\ &= \frac{x_0}{2^{l/2}} + \frac{F_0(x_0)}{2^{l/2}} + \frac{F_1(x_1)}{2^{(l-1)/2}} + \frac{F_2(x_2)}{2^{(l-2)/2}} + \cdots + \frac{F_{l-1}(x_{l-1})}{2^{1/2}} . \end{aligned}$$
>
>由此可以看出，Post-Norm 会导致梯度的指数衰减。
>
>相对的，在使用 Pre-Norm 的情况下迭代展开之后可以看到：
>
>$$x_l = x_0 + F_0(x_0) + F_1\!\left(\frac{x_1}{\sqrt{2}}\right) + F_2\!\left(\frac{x_2}{\sqrt{3}}\right) + \cdots + F_{l-1}\!\left(\frac{x_{l-1}}{\sqrt{l}}\right)$$
>
>每一条残差通道都是平权的，残差的作用会比Post Norm更加明显，所以它也更好优化。

这也是为什么 Post Norm 要用 warmup 的原因，warmup 学习率指的是训练一开始不要直接用目标学习率，而是从很小（甚至 0）逐步升到目标学习率，常见是“线性升温 N 步”，然后再进入正常的衰减策略。

Post Norm 在训练初期会导致靠近输出端的一些参数梯度在期望上偏大、层间梯度尺度也更不均衡，如果一上来就用较大的学习率，这些“大梯度 × 大步长”很容易把优化推到不稳定区间（震荡、发散、NaN）。Warmup 的作用就是先用小学习率让模型把激活/梯度尺度“驯化”到更可控的范围，再逐步升到目标学习率，从而避免 Post Norm 初期的不稳定问题。

我们接下来讨论这个很反直觉的问题：为什么 Pre Norm 这么好训练，效果反而没 Post Norm 那么好呢？

参考 [如何评价微软亚研院提出的把 Transformer 提升到了 1000 层的 DeepNet？](https://www.zhihu.com/question/519668254/answer/2371885202)- 唐翔昊的回答 ，残差链接会导致模型实际深度小于模型的层数，观察 $x_{t+1} = x_t+F_t(\text{Norm}(x_t))$ 可以发现因为 Normalization 操作的原因，第二项的方差是恒等不变的，但经过不断累加，$x$ 的方差会在主干上不断积累，层数高了以后单层对主干的影响是很小的。

我们可以看到：

$$
\begin{aligned}
x_{t+1} &= x_t + F(\operatorname{Norm}(x_t)) \\
        &= x_{t-1} + F(\operatorname{Norm}(x_{t-1})) + F(\operatorname{Norm}(x_t))
\end{aligned}
$$

当 $t-1$ 足够大，也就是模型层数足够深时，$F_{t-1}(\operatorname{Norm}(x_{t-1}))$ 和 $F_t(\operatorname{Norm}(x_t))$ 的区别是不大的，也就是说上述的式子约等于：

$$
x_{t-1} + 2F(\operatorname{Norm}(x_{t-1}))
$$

这就导致了深处的层发虚，模型实际深度小于模型的层数。

而 Post Norm 牺牲了从头到尾的恒等路径，保证了方差恒定，每一层对 $x$ 的影响都够大，保证了模型深度更真实，难收敛但寻出来的效果更好。

## 3. Layer Normalization 和 Batch Normalization

Batch Normalization：沿 Batch Size 维度，即对对同一批数据中相同位置的词向量做归一化。

Layer Normalization：沿 Embedding 维度，即对每个样本的所有特征做归一化。在训练样本较小、样本间相互影响较大的情况下更稳定。

LN（Layer Normalization）会先在“每个样本内部、指定的特征维度”上计算均值和方差：

$$
\begin{aligned}
\mu &= \frac{1}{d}\sum_{i=1}^{d} x_i\\

\sigma &= \sqrt{\frac{1}{d}\sum_{i=1}^{d} (x_i - \mu)^2}
\end{aligned}
$$

然后把该样本这组特征做标准化，$\epsilon$ 的作用是防止除零：

$$
X_{\text{norm}} = \frac{X - \mu}{\sigma + \epsilon}

$$

最后为了不损失表达能力，再做一次可学习的逐元素仿射变换，这些统计量来自“单个训练样本在同一层的所有单元”，不依赖 batch：

$$
Y = w \cdot X_{\text{norm}} + b

$$

代码实现：
```python
class LayerNorm(nn.Module):
    # x: (bsz, max_len, hidden_dim)
    def __init__(self, hidden_dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_dim))
        self.beta  = nn.Parameter(torch.zeros(hidden_dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        var  = x.var(dim = -1, keepdim = True, unbiased = False) 
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


bsz, max_len, hidden_dim = 2, 4, 8
x = torch.randn(bsz, max_len, hidden_dim)

LN = LayerNorm(hidden_dim)
y = LN(x)

print(x.shape, y.shape) 
```

为什么 nlp 任务中常用 Layer Normalization 而不是 Batch Normalization 呢？

首先 Batch Normalization 要求 Batch 足够大，因为 Batch Norm 需要跨样本，大 Batch 才能抓到样本的分布，而加大 Batch 后在多卡的环境下就需要额外通信开销。

其次 nlp 序列长度不固定，这样就导致 Batch 最后总会有 PAD 来占位，而 PAD 则会干扰数据分布，计算时忽略 PAD 又会降低 Batch 的大小。

最后，训练时用当前 mini-batch 的均值/方差做归一化、推理时改用训练期累计的 running mean/var（因为推理可能 batch=1），如果测试分布和训练分布偏移就会出现训练/预测不一致、性能变差。

## 4. RMSNorm（Root Mean Square Layer Normalization）

RMSNorm 可以看成是去掉了减均值步骤的 LayerNorm，通过舍弃中心不变性来降低计算量。

对每个样本的 hidden 向量  $x\in\mathbb{R}^n$ ，先算它的 RMS（均方根）：

$$
\operatorname{RMS}(x) = \sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2 + \epsilon}
$$

然后只做按 RMS 缩放的归一化并乘可学习缩放参数：

$$
y_i = \frac{x_i}{\operatorname{RMS}(x)} \cdot \gamma_i
$$

它的核心动机是：LayerNorm 里的去均值带来的 re-centering 不一定是必须的，去掉后计算更简单、更省时，同时在很多任务上能做到与 LayerNorm 相当的效果。原论文报告在不同模型上能减少约 7%–64% 的运行时间。

>re-centering 就是“把向量整体平移到零均值”：先在归一化维度上算均值 $\mu$，再做 $x \leftarrow x-\mu$（也就是 LayerNorm 里的“减均值”那一步）；RMSNorm 说的去掉 re-centering就是不做这步，只做按 RMS/尺度的归一化。

代码实现：
```python
class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        add_unit_offset: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + self.eps)

    def forward(self, x):
        x = self._norm(x.float()).type_as(x)
        if self.add_unit_offset:
            output = x * (1 + self.weight)
        else:
            output = x * self.weight
        return output
```