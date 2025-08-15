---
created: 2025-06-20
tags:
  - cs336
---

### 首先配置好 `vscode` 的 `settings.json`

因为 uv 项目会在工程目录下新建一个 `.venv` 在这里面放置各种环境依赖，而 vscode 进行文件解析（如果把解释器错误的设置为平时用的路径，那么大概率会出现各种不能成功导入包，出现各种报错, 这种错误我曾经一笑而过，但是也确实在潜意识里让我觉得很难受，直到上 jyy 的课时我知道了 vscode 是 `需要` 也是 `可以` 去设置类似解释器，类似调试器路径，类似等等等操作，不过后来还是不太知道该怎么设置，不过让我知道了可以这么设置，而大模型就完全能够让我正确设置 `settings.json`）

``` json
{
    "python.analysis.typeCheckingMode": "basic",
    "python.defaultInterpreterPath": "${workspaceFolder}/assignment1-basics/.venv/bin/python",      
}
```


### 对于大模型工具等声明

该声明是该 assignment 里的，强调了三点
1. 最好不要用各种大模型自动补全工具
2. 也没有大模型补全工具能够完成该作业
3. 使用大模型补全工具会让使用者不能深层地理解该作业


### 创建 $W$ 的技巧

手动实现一个 nn. Linear 层中的权重初始化逻辑，具体使用了一个叫 `truncated normal`（截断正态分布） 的初始化方式。我们来逐步解释这段代码都做了什么、为什么这么做，以及它的意义。




 #### 一、代码逐行解释

 定义可学习参数  $W$

``` python
Self.W = nn.Parameter (
    Torch.Empty (self.Out_features, self.In_features, device=self.Device, dtype=self.Dtype)
)
```


手动创建一个 `torch.nn.Parameter`，表示一个需要梯度的张量

明确控制初始化方式，这里是截断正态分布（Truncated Normal）

不使用 nn. Linear，可能需要自己写 forward (x @ self. W. T) 来完成前向传播

📌 优点：

高度可控（初始化方式、形状、设备、dtype 等）

适合写自定义的模块，如 Transformer、BERT 等论文级实现

📌 缺点：

更复杂、冗长

容易出错（需手动处理转置、广播等）


这行代码定义了一个参数张量 W，大小为 [输出维度, 输入维度]，用来存放线性层的权重（即权重矩阵 W）。
	•	用 torch.Empty () 只分配内存，不初始化值
	•	用 nn.Parameter () 表示它是一个可训练的参数

``` python
self.w1 = nn.Linear(d_model,d_ff,bias=False)
```

- 使用的是 PyTorch 提供的封装模块 nn. Linear

- 自动创建权重参数 `self.W1.Weight`（形状为\[d_ff, d_model]）

- 自动注册参数、添加到 `model.Parameters()` 中

- 可选 bias 参数，设为 False 即不添加偏置

- 默认使用 Kaiming Uniform 初始化权重

-----

2. 计算初始化标准差 std

$Std = 2 / (self. In_features + self. Out_features)^{0.5}$

这是 Xavier/Glorot 初始化中的一种变种，也叫 LeCun 初始化风格：

$\text{std} = \frac{2}{\sqrt{\text{fan\_in} + \text{fan\_out}}}$

这个标准差值控制了初始值分布的“宽度”，有助于在网络初始化时避免梯度爆炸或消失。


3. 使用截断正态分布初始化

``` python
Init.runc_normal_(self.W, std=std, a = -3 * std, b = 3 * std)
```

这表示：
	•	使用正态分布 𝒩(0, std²) 初始化权重
	•	但只取落在区间 \[$-3\cdot \text{std}, 3\cdot \text{std}$] 内的值
	•	超出范围的会重新采样，保证数值“集中”，避免极端值

这就是所谓的截断正态分布（`truncated normal`），相比直接 `torch.Randn()` 更安全。

⸻

#### 二、为什么要用 trunc_normal_ 初始化？

优点解释
避免极端值限定在 ±3σ 内，保证初始梯度稳定
有良好均值和方差控制保证正态分布，但去掉尾部异常值
更适合深层网络避免深层传播中梯度消失或爆炸

这在 Transformer 模型、ResNet、ViT 等中都很常见，很多模型用它作为默认权重初始化。


补充：`init.trunc_normal_` 是什么？

PyTorch 中：

`torch.nn.init.trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0)`

	•	在区间 [a, b] 内从正态分布采样
	•	mean 是均值，std 是标准差
	•	初始化结果仍近似于 𝒩(mean, std²)，但裁剪了尾部




当然有！在 PyTorch 中，有多种 **参数初始化方式**，可以用于 `nn.Parameter` 或 `nn.Linear.weight` 等张量。PyTorch 的初始化函数主要集中在 `torch.nn.init` 模块中。

---

### 🔧 常见初始化方法总览（`torch.nn.init`）

| 初始化方法                | 适用情况                    | 函数名                                    | 分布特性                              |
| -------------------- | ----------------------- | -------------------------------------- | --------------------------------- |
| **均匀初始化**            | 通用，简单                   | `uniform_`                             | 均匀分布 $[a, b]$                     |
| **正态初始化**            | 通用，简单                   | `normal_`                              | 正态分布 $\mathcal{N}(\mu, \sigma^2)$ |
| **截断正态初始化**          | 防止极端值（更稳定）              | `trunc_normal_`                        | 正态分布截断在 $[a, b]$                  |
| **Xavier 均匀/正态初始化**  | 前后层激活值方差相同，常用于前馈层       | `xavier_uniform_` / `xavier_normal_`   | 方差基于输入+输出维度                       |
| **Kaiming 均匀/正态初始化** | 常用于 ReLU 激活前的权重         | `kaiming_uniform_` / `kaiming_normal_` | 方差考虑非线性激活的影响                      |
| **常数初始化**            | 用于特殊需求（例如测试、调试）         | `constant_`                            | 所有元素初始化为同一个值                      |
| **单位初始化**            | 用于特殊结构如 RNN、残差连接等       | `eye_`                                 | 对角为 1，其余为 0（单位矩阵）                 |
| **稀疏初始化**            | 用于稀疏模型（如稀疏 Transformer） | `sparse_`                              | 指定稀疏率                             |
| **Zeros / Ones**     | 常用于偏置初始化                | `zeros_`, `ones_`                      | 所有元素为 0 或 1                       |

---

## 🎯 实例：几种典型初始化 $W$ 的方法

### 1. Xavier 初始化（适用于 Sigmoid/Tanh 激活）

```python
init.xavier_uniform_(self.W)
# or
init.xavier_normal_(self.W)
```

### 2. Kaiming 初始化（适用于 ReLU）

```python
init.kaiming_uniform_(self.W, a=0, mode='fan_in', nonlinearity='relu')
# or
init.kaiming_normal_(self.W, mode='fan_in', nonlinearity='relu')
```

### 3. 截断正态分布初始化（控制极端值）

```python
std = 0.02
init.trunc_normal_(self.W, mean=0.0, std=std, a=-2*std, b=2*std)
```

> GPT/BERT/Transformer 模型中，很多时候都使用 `trunc_normal_`，例如：

 ```python 
 Init.trunc_normal_(self.W, std=0.02)
  ```

### 4. 纯手动初始化

```python
self.W.data = torch.randn_like(self.W) * 0.01
```

---

## 🧠 如何选择初始化方式？

| 网络结构             | 建议初始化方式                          |
| ---------------- | -------------------------------- |
| MLP + ReLU       | `kaiming_uniform_`               |
| Transformer/BERT | `trunc_normal_(std=0.02)`        |
| Sigmoid/Tanh 网络  | `xavier_uniform_`                |
| RNN/LSTM         | `orthogonal_`, `xavier_uniform_` |
| 测试/调试模型          | `constant_`, `ones_`, `zeros_`   |


> **初始化影响模型训练稳定性与收敛速度，权重初始化是深度学习中提升模型性能的“隐形手术刀”。**




## `trick`



根据这个 Transformer 作业的文档，我来解释为什么有些模块使用 class，有些使用 function：

## 使用 Class 的模块

**需要可学习参数的组件：**

1. **Linear** - 有权重矩阵 W 参数
2. **Embedding** - 有嵌入矩阵参数  
3. **RMSNorm** - 有可学习的增益参数 g
4. **PositionWiseFeedForward** - 包含多个 Linear 层
5. **MultiHeadSelfAttention** - 有 $W_Q$,$ $W_K$, $W_V$, $W_O$ 参数
6. **TransformerBlock** - 组合多个有参数的子模块
7. **TransformerLM** - 整个模型的顶层容器

这些都继承自 `torch.nn.Module`，因为它们：
- 包含需要训练的参数（`nn.Parameter`）
- 需要参与梯度计算和反向传播
- 需要状态管理（如设备、数据类型）
- 需要序列化/反序列化功能

## 使用 Function 的模块

**无可学习参数的操作：**

1. **Softmax** - 纯数学运算，无参数
2. **Scaled Dot-Product Attention** - 计算注意力权重，无参数
3. **RoPE（可选）** - 位置编码，使用预计算的 sin/cos 值

这些实现为 function 因为它们：
- 不需要可学习参数
- 是无状态的数学运算
- 可以更高效地实现
- 便于复用和测试

## 设计原则

```python
# Class - 有参数
class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
    
    def forward(self, x):
        return x @ self.weight.T

# Function - 无参数
def softmax(x, dim):
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x_shifted = x - x_max
    return torch.exp(x_shifted) / torch.sum(torch.exp(x_shifted), dim=dim, keepdim=True)
```

这种设计遵循了 PyTorch 的最佳实践：**有状态/参数的用 Module，无状态的用 Function**。

