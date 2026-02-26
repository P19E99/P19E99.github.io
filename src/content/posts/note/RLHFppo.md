---
title: RLHF（一）：PPO 基本流程
published: 2026-02-26
description: "PPO 基本流程"
image: "./cover2.jpg"
tags: ["Interview Notes"]
category: Interview Notes
draft: false
---


## 1. RLHF（Reinforcement Learning from Human Feedback）

为什么在 SFT 之后我们需要 RLHF？

个人理解是，SFT 更偏向于对下游任务的适配，而 RLHF，他叫 Reinforcement Learning from Human Feedback，顾名思义是基于人类偏好的一种对齐。

比如模型知道 “114514” 和 “一一四五一四”，但明显 “114514” 易读一点，RLHF 就是让模型在面对输出 “114514” 和 “一一四五一四” 的选择时选择 “114514”。

预训练让模型变聪明，RLHF 让模型变听话、好用、靠谱一点。它主要解决的是“两个答案都不算错，但哪个更符合人类预期”这个问题，比如更有条理、少废话、少乱编、语气更合适。

## 2. LLM 环境下的 RL

我们知道在强化学习中有智能体、环境、状态、动作等等，这些在 LLM 中指什么呢？

思考第一部分我们做 RLHF 的目的，输入 prompt，模型生成符合人类喜好的 response。

那我们可以很自然的想到，在 LLM 中：
* 智能体对应的就是模型本身
* 动作就是就是模型生成新的 token
* 环境就是奖励模型或人类，智能体做出动作（生成 token）后给出反馈
* 状态就是某时刻的上文，比如时刻 $t$ 生成新的 token，状态由 $S_t$ 变为 $s_{t+1}$ 

## 3. RLHF 中的 PPO

RLHF 里的 PPO 就是用奖励模型给的偏好分数来更新语言模型的生成策略。先让模型对一批 prompt 生成回答，再用奖励模型打分，并加上对参考模型的 KL 惩罚把别跑偏写进目标里，然后用 PPO 的 clipped 更新把高分回答对应的 token 概率拉高、低分的拉低，从而让模型更符合人类偏好且训练过程稳定不发散。

## 4. PPO 组件

在进行 PPO 之前，我们需要准备四个模型：
1. **Policy $\pi_\theta$：** 也就是我们要训练的模型。
2. **Critic $V_\psi$：** 用来算优势，也是训练的一部分。
3. **Reference $\pi_{\text{ref}}$：** 这个模型参数是冻结的，为了让模型的输出不跑偏（算KL）。
4. **Reward $r_\phi$：** 参数冻结，给偏好分。  

### 4.1 Policy Model

这个就是要训练的目标模型，即强化学习中负责执行动作的部分。它根据当前的状态（上下文）选择动作（生成 token），并将动作发送给环境。

### 4.2 Critic Model

大部分情况下和 Policy Model 用同一个主干，改造 Policy Model，在顶上再加一个 value head。policy head 输出 logits，value head 输出 value。

### 4.3 Reference Model

负责提供 KL 约束的参照系，防止 Policy 为了刷奖励模型分数而走到语言崩坏/模式崩坏/胡说八道的区域。通常把初始的 Policy Model 拷一份，然后冻结参数。

### 4.4 Reward Model
用于计算生成 token 的即时收益，在 RLHF 过程中，它的参数是冻结的。

## 5. PPO 流程

在 PPO 之前，我们可以先训个 Reward Model，这一步是可选的。

第一步，Policy 生成答案（rollout）。从数据集取一批 prompts $x$。用当前 Policy $\pi_\theta$ 对每个 $x$ 生成回答 token 序列 $y=(a_1,\dots,a_T)$，并记录每步的 $\log \pi_\theta(a_t\mid x,a_{<t})$。同时把生成这一批数据时的策略记作 $\pi_{\theta_{\text{old}}}$。

第二步，Reward 打分。把每个 $(x,y)$ 丢给奖励模型，得到标量分数 $r_\phi(x,y)$。这就是人类偏好信号的代理。

第三步，Reference 算 KL 惩罚。同一个生成序列上，用 Reference 计算 $\log \pi_{\text{ref}}(a_t\mid x,a_{<t})$，和 Policy 的 logprob 做差，构造 KL 惩罚项，再把它和奖励模型分数组合成总回报。直觉就是：分数高很好，但如果是靠把模型改得面目全非换来的，也要扣分。

第四步，Critic 估计回报并算优势 $A_t$。用 Critic 在每个状态 $s_t=(x,a_{<t})$ 上预测 $V_\psi(s_t)$，再用总回报（奖励分 + KL 惩罚，通常把句末主奖励和逐 token KL 惩罚结合）算出每个 token 的优势 $A_t$。优势的含义很简单：在这个前缀下选了当前 token，到底对最终高分是正贡献还是负贡献，贡献多大。

第五步，PPO 更新 Policy。对每个 token 计算概率比：

$$
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}
$$

如果 $A_t>0$，希望提高这个 token 概率，如果 $A_t<0$，希望降低。但 PPO 用 clip 把 $r_t(\theta)$ 限在 $[1-\epsilon,1+\epsilon]$ 附近，避免一次更新过猛。然后用这个 clipped 目标做梯度下降更新 $\theta$。同一批 rollout 往往会做几轮 minibatch/epoch 更新，提高样本利用率。

> clip 就是把一个数强行限制在某个区间里，超出就截断到边界。
> 在 PPO 里，clip 作用在“概率比”上。先定义概率比：
> 
> $$
> r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\text{old}}(a_t\mid s_t)}
> $$
> 
> 然后做截断：
> $$
> \mathrm{clip}(r_t,,1-\epsilon,,1+\epsilon)=\begin{cases}1-\epsilon, & r_t<1-\epsilon \\r_t, & 1-\epsilon \le r_t \le 1+\epsilon \\1+\epsilon, & r_t>1+\epsilon\end{cases}
> $$
> 
> PPO 的核心目标把原来的更新和截断后的更新取最小也就是更保守的那个：
> $$
> L^{\mathrm{CLIP}}(\theta)=\mathbb{E}_t\Bigl[\min\bigl(r_t A_t,\ \mathrm{clip}(r_t,1-\epsilon,1+\epsilon)A_t\bigr)\Bigr]
> $$

第六步，同时更新 Critic。用回报去拟合 $V_\psi(s_t)$，更新 (\psi)，让优势估计更准。

第七步，循环。用更新后的 Policy 再去生成新答案，重复上述步骤。训练过程中会监控 KL 大小并调 $\beta$（KL 的惩罚系数，越大 KL 扣分越重），让 Policy 既能涨分，又不至于偏离 Reference 太远。

-----

这期先写到这里，下一期会写代码方面的解析。