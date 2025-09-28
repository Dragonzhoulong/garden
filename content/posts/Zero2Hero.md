---
title: " Zero2Hero "
date: " 2025-08-25"
tags:
  - llm
draft: false
---


来自 Karpathy 的系列视频。对于代码实现容易出错的地方讲解的非常细致，但是对于数学理论的推导引入不足，但是这样一位在深度学习领域工作十余年的大能来说，可以为听者提供非常宽广的视野，有点已识乾坤大，犹怜草木青
# Bigrams

https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part1_bigrams.ipynb

https://youtu.be/VMj-3S1tku0?si=e9qLGnaQ1Img3iBL

上面是系列视频与视频中用到的代码文件

## 要点

### `sum` 函数的 `keepdims`

对于 `torch` 的 `sum` 来说默认 `keepdims` 是 `false` 的，这样在 1 维度上累加之后会把维度折叠，导致传播机制的除法出现问题（不是作为列向量去每列除，而是作为行向量每行除）

``` python

P.sum(1).shape
torch.Size([27])
# 27, 27
#  1, 27

P = (N+1).float()
P /= P.sum(1, keepdims=True)

# 27,27
# 27,1

```

### `p = p / p.sum()` 不是原地修改

``` python

g = torch.Generator().manual_seed(2147483647)
p = torch.rand(3, generator=g)
p = p / p.sum()



P = (N+1).float()
P /= P.sum(1, keepdims=True)

```

这两个操作在语义上很相近，但在 **Python / NumPy / PyTorch** 等框架中，它们之间 **确实存在关键差别** ——主要在于是否是 **原地（in-place）操作**，以及是否会影响原变量 `p` 的存储、梯度等。

---

#### 🧮 一行解释区别

| 表达式               | 是否原地操作 | 含义                          |
| ----------------- | ------ | --------------------------- |
| `p = p / p.sum()` | ❌ 否    | 创建新 tensor，并用它替换变量 `p`（非原地） |
| `p /= p.sum()`    | ✅ 是    | 原地修改 `p`，即在原有内存上直接更新值       |

---

#### 🧠 更深入的理解（以 PyTorch 举例）

#### 1. `p = p / p.sum()`

* **发生了新 tensor 的创建**：
  `p / p.sum()` 会先计算一个新的 tensor，`p` 原有的 tensor 不变。
* 然后再把变量名 `p` 绑定到这个新的 tensor 上。
* 如果你之前有代码依赖 `p` 的旧值（或它的内存地址），它已经变了。

#### 2. `p /= p.sum()` （等价于 `p = p.__itruediv__(p.sum())`）

* **原地修改**：直接在 `p` 的内存上进行除法更新。
* 节省内存，性能可能更好。
* 在 PyTorch 中，如果 `p.requires_grad=True`，**原地操作可能会报错或影响 autograd 的正确性**，因为它破坏了计算图中对中间变量的追踪。

---

#### 🧪 举个简单的 PyTorch 例子

```python
import torch
p = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# 非原地操作
q = p
p = p / p.sum()
print(p is q)  # False

# 原地操作
p = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
q = p
p /= p.sum()
print(p is q)  # True
```

---

#### 🧨 实战注意（特别是 PyTorch）

* **不要在 autograd 追踪中随意用原地操作**（如 `/=`, `+=`, `relu_()`），因为会导致梯度追踪错误或报错。
* 推荐默认使用非原地操作（`p = p / p.sum()`），除非你非常确定需要手动优化内存或性能，并能正确处理 autograd 的行为。

---

#### ✅ 总结

| 表达式               | 是否推荐    | 原因                   |
| ----------------- | ------- | -------------------- |
| `p = p / p.sum()` | ✅ 推荐    | 安全，不影响 autograd，行为明确 |
| `p /= p.sum()`    | ⚠️ 小心使用 | 是原地操作，可能带来效率但也带来风险   |

需要我用 PyTorch 的 autograd 举一个“原地操作出错”的例子也可以继续说。
