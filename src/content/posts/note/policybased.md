---
title: RLHF理论基础：策略梯度
published: 2026-02-22
description: "策略梯度"
image: "./cover4.jpg"
tags: ["Interview Notes"]
category: Interview Notes
draft: false
---

## 1. 为什么需要策略梯度

为什么我们需要 Policy-based，Value-based的局限在哪？

Value-based 核心做法是先学一个价值函数 $Q(s,a)$，再用 $\arg\max_a Q(s,a)$ 选动作。它的局限主要在某些场景里会变得很别扭。

一个是动作空间一旦是连续的或很大的离散空间，$\arg\max_a Q(s,a)$ 就成了硬伤。离散动作还可以枚举，但动作维度一高就爆了。比如机器人的抬手角度就不是一个离散的值，我们没法对角度进行打分。

另一个是 Value-based 像先给每个动作打分，再选最高分，Policy-based 像直接学怎么出手，以及出手的概率。问题出在现实里不一定总是选唯一最高分这么简单。Value-based 本质上是一个确定性策略，比如在石头剪刀布的时候如果出石头赢过几次，石头的分就会很高，就会一直出石头。但是我们都知道最佳策略是各占 $\frac{1}{3}$ 随机出。

而 Policy-based：不学打分直接学怎么做。他天然支持连续动作、策略直接、能随机、更适合复杂真实任务。

## 2. Policy-based

Policy-based 直接优化策略，不再学习值函数去打分。

我们将策略参数化写成：

$$
\pi_\theta(a|s)
$$

意思是在在状态 $s$ 下，采取动作 $a$ 的概率是多少。也就是说，策略网络输出的不是一个分数，而是一整个动作概率分布。

然后定义一个目标函数 

$$
J(\theta)
$$

表示当前策略 $\pi_\theta(a|s)$ 的表现（通常是期望累计回报），然后沿着让 $J(\theta)$ 变大的方向更新参数。

## 3.策略梯度

与让损失函数下降的梯度下降方法相反，我们想让 $J(\theta)$ 变大就可以使用梯度上升的方法，这个就叫策略梯度。

OK，既然知道了我们的方法，我们定义我们的 $J(\theta)$ 为：

$$
J(\theta)=\mathbb{E}_{\tau\sim \pi_\theta}[R(\tau)]
$$

按当前策略 $\pi_\theta$ 去跑轨迹 $\tau$，得到一个总回报 $R(\tau)$，$J(\theta)$ 就是这些总回报期望。

参考 东川路第一可爱猫猫虫 的公式可能更好理解一些（这里致谢一下）：

$$
J(\theta) = \sum_{\tau} P(\tau;\theta)\, R(\tau)
$$

也就是说采用 $\theta$ 策略时走轨迹 $\tau$ 的概率乘上走轨迹 $\tau$ 的累加奖励求和，其实跟上面是一样的。

要做梯度上升我们要先对他求个梯度：

$$
\begin{aligned}
\nabla_{\theta} J(\pi_{\theta})
&= \nabla_{\theta}\,\mathbb{E}_{\tau \sim \pi_{\theta}}[R(\tau)] \\
&= \nabla_{\theta} \int_{\tau} P(\tau \mid \theta)\,R(\tau) \\
&= \int_{\tau} \nabla_{\theta} P(\tau \mid \theta)\,R(\tau) \\
&= \int_{\tau} P(\tau \mid \theta)\,\nabla_{\theta}\log P(\tau \mid \theta)\,R(\tau) \\
&= \mathbb{E}_{\tau \sim \pi_{\theta}}\!\left[\nabla_{\theta}\log P(\tau \mid \theta)\,R(\tau)\right]
\end{aligned}
$$

$$
\therefore\;
\nabla_{\theta} J(\pi_{\theta})
=
\mathbb{E}_{\tau \sim \pi_{\theta}}\!\left[
\sum_{t=0}^{T}
\nabla_{\theta}\log \pi_{\theta}(a_t \mid s_t)\,R(\tau)
\right]
$$

## 4. Reinforce算法（蒙特卡罗策略梯度）

我们有了梯度，就可以根据这个梯度做梯度上升：

$$
\nabla_{\theta} J(\pi_{\theta})
=
\mathbb{E}_{\tau \sim \pi_{\theta}}\!\left[
\sum_{t=0}^{T}
\nabla_{\theta}\log \pi_{\theta}(a_t \mid s_t)\,R(\tau)
\right]
$$

上面的梯度是一个期望，实际中我们用当前策略跑若干条轨迹，用样本均值近似这个期望：

$$
\hat{g}
=
\frac{1}{N}
\sum_{i=1}^{N}
\sum_{t=0}^{T_i}
\nabla_{\theta}\log \pi_{\theta}\!\left(a_t^{(i)} \mid s_t^{(i)}\right)\, R(\tau_i)
$$

然后按梯度上升更新参数:

$$
\theta \leftarrow \theta + \alpha \hat{g}
$$

直到收敛。