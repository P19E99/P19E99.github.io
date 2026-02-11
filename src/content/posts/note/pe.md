---
title: 【八股】位置编码（Positional Encoding）
published: 2026-02-11
description: "Positional Encoding"
image: "./cover2.jpg"
tags: ["dl"]
category: dl
draft: false
---
## 1. 为什么需要位置编码

从人类的角度来看，我们对于一整排的输入很容易知道他们的位置关系，但是在 Transformer 里，注意力本质上只看“token 与 token 的相似度”，如果你把一句话里 token 的顺序打乱、但对应的向量集合不变，纯 self-attention 很难仅靠结构本身判断“谁在前谁在后”。比如”我吃鱼“和”鱼吃我“，在 self-attention 的眼中没有任何区别。

Transformer（没有 RNN/CNN 的顺序归纳偏置）对序列是近似“置换不变”的，所以必须额外注入位置信息，这就是我们为什么需要位置编码。

## 2. 正/余弦位置编码”（Sinusoidal Positional Encoding）

这是《Attention Is All You Need》里 Transformer 所使用的位置编码，是绝对位置编码，论文里做法是：给序列中每个位置 pos 生成一个与词向量同维度 $d_{\text{model}}$ 的位置向量，然后与 token embedding 逐元素相加，然后把这个“embedding+position”的和送进 Encoder/Decoder 的第一层。

它的定义是（i 是维度索引）：
$$
PE(pos, 2i)=\sin\Big(\frac{pos}{10000^{2i/d_{\text{model}}}}\Big),\quad
PE(pos, 2i+1)=\cos\Big(\frac{pos}{10000^{2i/d_{\text{model}}}}\Big)
$$
也就是偶数维用 sin，奇数维用 cos；每一对维度 $(2i,\ 2i+1)$ 对应一个不同频率（不同“波长”）的正弦信号。

这种绝对位置编码，对固定偏移 $k$，每一对维度满足加法公式，$PE(pos+k)$ 可以由 $PE(pos)$ 通过一个只依赖 $k$ 的线性变换得到。直观上，这让模型更容易从绝对坐标里“还原”相对距离/位移模式（例如对齐、复制、邻近依赖）。

但这种编码方式外推性差，练没见过的位置就没有对应向量；而 sin/cos 虽然理论上可外推到任意 pos，但在实践里，很多工作发现它在更长上下文上的泛化不如专门为外推设计的相对/偏置类方法

## 3. ALiBi（Attention with Linear Biases）

ALiBi 是一个把位置信息塞进注意力分数里的方法，目标是让 Transformer 长度外推更稳。它的关键点在于，不再把位置向量加到 token embedding 上；而是在每个注意力 head 的 $QK^⊤$ 分数上，加一个与距离成线性关系的偏置。

标准注意力的 logits 是：
$$
S_{i,\ j}=\frac{Q_iK^⊤_j}{\sqrt d}
$$
在 ALiBi 中把他改成：
$$
S'_{i,\ j}=\frac{Q_iK^⊤_j}{\sqrt d}+b_h(i,\ j)
$$
其中 $b_h(i,j)$ 是第 $h$ 个 head 的线性距离偏置。对常见的 decoder-only、causal attention 来说，一般写成：
$$
b_h(i,j) = -m_h(i-j) \quad (j \le i)
$$

意思是：key 越远（$i-j$ 越大），分数被扣得越多；所以 softmax 后天然更偏向近处 token。可以把它理解为注意力在内容相似度之外，还额外背了一个很简单的先验：离我越远，我越不信。这个先验是通过给 logits 加线性惩罚实现的。

对于 $m_h$ ，不同 head 用不同的斜率 $m_h$，形成一组几何级数。直觉上，有些 head 惩罚很陡，几乎只看很近的上下文；有些 head 惩罚很缓，可以看得更远。[论文](https://arxiv.org/pdf/2108.12409)给了一个固定的构造规则，常见实现会按 head 数的几何分布生成。

ALiBi 的优点在于参数为 $0$ ，只需要在 attention logits 上加一个矩阵，计算开销低，且对外推友好，偏置只和距离有关，长度变长时规则自然延伸。

缺点在于它偏向近的更重要，对需要强依赖超远距离（例如必须回看很前面的关键信息）的任务，可能会有天然劣势。

## 4. 旋转位置编码（RoPE，Rotary Position Embedding）

RoPE 通过将词向量在多维空间中“旋转”，并且旋转的角度与单词的绝对位置相关，使得两个向量在经过这种位置编码后，它们的点积只与它们之间的相对位置差有关，而与它们的绝对位置无关。

假设 $q_m$ 是位置 $m$ 的 token 的 $q$ ， $k_n$ 是位置 $n$ 的 token 的 $k$ ，我们将词嵌入向量与绝对位置信息相乘（也就是进行旋转）得到 $q_me^{im\theta},\ k_ne^{in\theta}$  ，根据欧拉公式 $\displaystyle e^{ix}=\cos x+i\sin x$ 可知乘以 $e^{ix}$ 相当于在复数平面逆时针旋转 $x$ 。

此时来看下两个token间的关注度，也就是 $q_me^{im\theta},\ k_ne^{in\theta}$ 做一个内积，即：
$$
\begin{aligned}
\langle q_m e^{im\theta},\ k_n e^{in\theta}\rangle
&= (q_m e^{im\theta})^*(k_n e^{in\theta})\\
&= q_m^* e^{-im\theta}, k_n e^{in\theta}\\
&= q_m^* k_n, e^{i(n-m)\theta}
\end{aligned}
$$

可以发现我们最后的结果只与 $q,\ k,\ (m-n)$ 相关，也就是说我们通过绝对位置信息添加到词嵌入向量里，得到了 token 间的相对位置信息。

抛开复数推导，我们还有一种形式理解 RoPE。

在二维特征空间下，我们可以对向量 $q_m$ 和 $k_n$ 乘一个逆时针正交旋转矩阵来让他进行逆时针旋转逆时针，其中逆时针正交旋转矩阵为：
$$
R_i =
\begin{pmatrix}
\cos i\theta & -\sin i\theta \\
\sin i\theta & \cos i\theta
\end{pmatrix}
$$

则分数计算部分则有：
$$
\begin{aligned}
(R_m q_m)^T (R_n k_n)
&=
\left[
\begin{pmatrix}
\cos m\theta & -\sin m\theta \\
\sin m\theta & \cos m\theta
\end{pmatrix}
\begin{pmatrix}
q_0 \\ q_1
\end{pmatrix}
\right]^T
\left[
\begin{pmatrix}
\cos n\theta & -\sin n\theta \\
\sin n\theta & \cos n\theta
\end{pmatrix}
\begin{pmatrix}
k_0 \\ k_1
\end{pmatrix}
\right]
\\
&=
\begin{pmatrix}
q_0 & q_1
\end{pmatrix}
\begin{pmatrix}
\cos((n-m)\theta) & -\sin((n-m)\theta) \\
\sin((n-m)\theta) & \cos((n-m)\theta)
\end{pmatrix}
\begin{pmatrix}
k_0 \\ k_1
\end{pmatrix}
\\
&=
q_m^T R_{n-m} k_n
\end{aligned}
$$

可以发现我们最后的结果只与 $q,\ k,\ (m-n)$ 相关，和上面得到了相同的结论。

以上是二维向量的推导，但实际应用中 token 向量的维度往往很高，这时我们可以把高维向量拆成若干个“对”，上面公式推导中的 $\theta$ 看作旋转的角速度，对于低维的“对”用较快的角速度，高位的“对”用较慢的角速度，所以我们设：
$$
\theta_i = 10000^{-2i/d},\ i = 0,1,2,\ldots,\frac{d}{2}-1
$$

因此在高维度空间中，用于表示 $q_m$ 绝对位置的旋转矩阵 $R_m$ 最终可以表示成：
$$
R_m q
=
\begin{pmatrix}
\begin{pmatrix}
\cos(m\theta_0) & -\sin(m\theta_0) \\
\sin(m\theta_0) & \cos(m\theta_0)
\end{pmatrix}
& \cdots & 0 \\
\vdots & \ddots & \vdots \\
0 & \cdots &
\begin{pmatrix}
\cos(m\theta_{d/2-1}) & -\sin(m\theta_{d/2-1}) \\
\sin(m\theta_{d/2-1}) & \cos(m\theta_{d/2-1})
\end{pmatrix}
\end{pmatrix}
\begin{pmatrix}
q_0 \\ q_1 \\ \vdots \\ q_{d-2} \\ q_{d-1}
\end{pmatrix}
$$

不过这是个稀疏矩阵，里面有大量的0，在实际应用中使用下面的方式：
$$
\begin{pmatrix}
q_0\\
q_1\\
q_2\\
q_3\\
\vdots\\
q_{d-2}\\
q_{d-1}
\end{pmatrix}
\otimes
\begin{pmatrix}
\cos m\theta_0\\
\cos m\theta_0\\
\cos m\theta_1\\
\cos m\theta_1\\
\vdots\\
\cos m\theta_{d/2-1}\\
\cos m\theta_{d/2-1}
\end{pmatrix}
+
\begin{pmatrix}
-q_1\\
q_0\\
-q_3\\
q_2\\
\vdots\\
-q_{d-1}\\
q_{d-2}
\end{pmatrix}
\otimes
\begin{pmatrix}
\sin m\theta_0\\
\sin m\theta_0\\
\sin m\theta_1\\
\sin m\theta_1\\
\vdots\\
\sin m\theta_{d/2-1}\\
\sin m\theta_{d/2-1}
\end{pmatrix}
$$