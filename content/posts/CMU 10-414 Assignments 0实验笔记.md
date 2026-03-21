---
title: " CMU 10-414 Assignments 实验笔记"
date: " 2025-12-02"
tags:
  - llm
  - python
draft:
---

来自于周鑫的非常好的笔记，非常全
https://www.zhouxin.space/notes/notes-on-cmu-10-414-assignments/#parse_mnist

在看完 `karpathy` 的 `micrograd` 视频并且手动实现完自动微分框架后，我想起来陈天奇的 deep learning system 课中也是关于实现一个深度学习框架介绍，于是我开始了学习 ` CMU 10-414`，希望学到更多的 python/numpy/cuda 编程技巧和更多的算法思想。



> **“What I cannot create, I do not understand.”** — Richard Feynman


# Parse_mnist

##  mnist 的解析
你这个函数就是要**从 MNIST 的原始二进制文件（gz 压缩的）里，把图片和标签读出来，变成 Numpy 数组**。
我给你分步骤讲清楚，再给一份可以直接用的实现代码。

---

## 一、先弄明白：MNIST 文件长什么样？

官网里说得比较抽象，我们简单化一下：

### 1. 图片文件（比如 `train-images-idx3-ubyte.gz`）

解压后是一个二进制文件，结构是：

1. 前 4 字节：magic number = **2051**
2. 接下来 4 字节：图片数量 `num_images`
3. 接下来 4 字节：每张图片的行数 `rows`（MNIST 是 28）
4. 接下来 4 字节：每张图片的列数 `cols`（MNIST 是 28）
5. 剩下的：`num_images * rows * cols` 个 **无符号字节**（`uint8`），每个值在 0~255，按顺序拼成所有图片的数据。

> 这些整数都是 **大端序**（big-endian），所以要用 `'>IIII'` 来解析。

### 2. 标签文件（比如 `train-labels-idx1-ubyte.gz`）

结构类似，但简单一些：

1. 前 4 字节：magic number = **2049**
2. 接下来 4 字节：标签数量 `num_labels`
3. 剩下的是 `num_labels` 个字节，每个标签是 0~9 的 `uint8`

---

## 二、函数需要返回什么？

根据你的 docstring：

* `X`:

  * 形状：`(num_examples, input_dim)`

    * 对 MNIST 就是 `(num_examples, 784)`，因为 28×28=784
  * 类型：`np.float32`
  * 数值范围：**0.0 ~ 1.0**

    * 所以要把原始 0~255 的 `uint8` 除以 255.0

* `y`:

  * 形状：`(num_examples,)`
  * 类型：`np.uint8`
  * 值是 0~9 的标签

并且需要保证：**图片数量和标签数量相同**（最好断言一下）。

---

## 三、实现的关键步骤

1. **导入需要的库**

```python
import gzip
import struct
import numpy as np
```

2. **读取图片文件**

* 用 `gzip.open(image_filename, 'rb')` 打开压缩文件
* 先读 16 字节作为头部

  * 用 `struct.unpack('>IIII', header)` 解析为 4 个无符号大端整数：

    * magic, num_images, rows, cols
* 用 `f.read()` 读剩下的所有字节

  * 用 `np.frombuffer(..., dtype=np.uint8)` 变成一维数组
  * `reshape(num_images, rows * cols)` 得到 `(num_images, 784)`
  * 转成 `np.float32` 并除以 `255.0`

3. **读取标签文件**

* 用 `gzip.open(label_filename, 'rb')`
* 先读 8 字节

  * 用 `struct.unpack('>II', header)` 得到：magic, num_labels
* 剩下的数据用 `np.frombuffer(..., dtype=np.uint8)` 直接变成 `(num_labels,)`

4. **做一些 sanity check（可选但推荐）**

* 检查 magic number 是否正确

  * 图片：2051
  * 标签：2049
* 检查 `num_images == num_labels`，不相等就报错

5. **返回 `(X, y)`**

---

## 四、完整实现示例

你可以直接把这段代码填到 `### BEGIN YOUR CODE` 和 `### END YOUR CODE` 之间：

```python
import gzip
import struct
import numpy as np

def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format. """

    # ---------- 1. 读图片 ----------
    with gzip.open(image_filename, 'rb') as f_img:
        # 1.1 读图片头部：magic, num_images, rows, cols
        header = f_img.read(16)
        magic, num_images, rows, cols = struct.unpack('>IIII', header)

        # 检查 magic number 是否正确（2051）
        if magic != 2051:
            raise ValueError(f'Invalid magic number in image file: {magic}')

        # 1.2 读取剩余的所有像素数据
        # 剩余长度应该是 num_images * rows * cols 字节
        img_data = f_img.read()
        images = np.frombuffer(img_data, dtype=np.uint8)

        # 1.3 reshape 成 (num_images, rows * cols)
        images = images.reshape(num_images, rows * cols)

        # 1.4 转成 float32，并归一化到 [0.0, 1.0]
        X = images.astype(np.float32) / 255.0

    # ---------- 2. 读标签 ----------
    with gzip.open(label_filename, 'rb') as f_lbl:
        # 2.1 读标签头部：magic, num_labels
        header = f_lbl.read(8)
        magic, num_labels = struct.unpack('>II', header)

        # 检查 magic number 是否正确（2049）
        if magic != 2049:
            raise ValueError(f'Invalid magic number in label file: {magic}')

        # 2.2 读取剩余的所有标签字节
        lbl_data = f_lbl.read()
        y = np.frombuffer(lbl_data, dtype=np.uint8)

    # ---------- 3. 一些安全检查 ----------
    if num_images != num_labels:
        raise ValueError(
            f'Image count ({num_images}) does not match label count ({num_labels})'
        )

    return X, y
```

---

## 五、再串起来总结一下实现逻辑

1. **用 gzip 打开文件**，因为原始的 MNIST 文件是 `.gz` 压缩的。
2. **先读固定长度的头部**（图片 16 字节，标签 8 字节），用 `struct.unpack` 按大端序解析成整数。
3. **读剩余所有数据**，直接用 `np.frombuffer` 转成 `uint8` 的一维数组。
4. 对图片：

   * 根据 `num_images, rows, cols` reshape 成二维 `(num_images, rows*cols)`。
   * 转成 `float32`，再除以 255.0 做归一化。
5. 对标签：

   * 直接保持为一维 `uint8` 数组。
6. 做检查：magic number 是否正确、图片数和标签数是否相等。
7. 返回 `(X, y)`。

好，我们这次不讲 MNIST 原理，只拆代码里的 **Python 语法**，一块一块解释 👍

先把实现贴一下，方便对照（跟我上次给你的是一样的）：

```python
import gzip
import struct
import numpy as np

def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format. """

    # ---------- 1. 读图片 ----------
    with gzip.open(image_filename, 'rb') as f_img:
        # 1.1 读图片头部：magic, num_images, rows, cols
        header = f_img.read(16)
        magic, num_images, rows, cols = struct.unpack('>IIII', header)

        # 检查 magic number 是否正确（2051）
        if magic != 2051:
            raise ValueError(f'Invalid magic number in image file: {magic}')

        # 1.2 读取剩余的所有像素数据
        img_data = f_img.read()
        images = np.frombuffer(img_data, dtype=np.uint8)

        # 1.3 reshape 成 (num_images, rows * cols)
        images = images.reshape(num_images, rows * cols)

        # 1.4 转成 float32，并归一化到 [0.0, 1.0]
        X = images.astype(np.float32) / 255.0

    # ---------- 2. 读标签 ----------
    with gzip.open(label_filename, 'rb') as f_lbl:
        # 2.1 读标签头部：magic, num_labels
        header = f_lbl.read(8)
        magic, num_labels = struct.unpack('>II', header)

        # 检查 magic number 是否正确（2049）
        if magic != 2049:
            raise ValueError(f'Invalid magic number in label file: {magic}')

        # 2.2 读取剩余的所有标签字节
        lbl_data = f_lbl.read()
        y = np.frombuffer(lbl_data, dtype=np.uint8)

    # ---------- 3. 一些安全检查 ----------
    if num_images != num_labels:
        raise ValueError(
            f'Image count ({num_images}) does not match label count ({num_labels})'
        )

    return X, y
```

---

## 1. `import` 语句

```python
import gzip
import struct
import numpy as np
```

* `import 模块名`：导入一个模块，这样可以使用里面的函数、类。
* `import numpy as np`：

  * `as np` 是给模块起一个别名，后面用 `np.xxx` 比写全称 `numpy.xxx` 更方便。

---

## 2. 函数定义和参数

```python
def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format. """
```

* `def 函数名(参数1, 参数2):`：定义一个函数。
* `image_filename`, `label_filename`：

  * 是两个形参（参数名），调用时会传具体的字符串文件名。
* 三引号 `""" ... """`：

  * 这是**文档字符串**（docstring），用来说明函数作用，不影响运行。
  * 写在函数第一行，`help(parse_mnist)` 时能看到。

---

## 3. `with` 上下文管理器

```python
with gzip.open(image_filename, 'rb') as f_img:
    ...
```

解释：

* `with 表达式 as 变量:` 是 **上下文管理器语法**：

  * 帮你自动做“打开 → 使用 → 关闭”资源（比如文件、网络连接），就算中间报错也会正确关闭。
* `gzip.open(image_filename, 'rb')`：

  * 调用 `gzip` 模块里的 `open` 函数。
  * `'rb'`：**r**ead **b**inary，二进制读。
* `as f_img`：

  * 把打开的文件对象绑定到变量 `f_img` 上，在 `with` 块里面用 `f_img` 操作文件。
* `with` 缗块结束后（缩进结束），文件会**自动关闭**，不需要手动 `f_img.close()`。

同理：

```python
with gzip.open(label_filename, 'rb') as f_lbl:
    ...
```

---

## 4. 赋值与多重赋值、解包

```python
header = f_img.read(16)
magic, num_images, rows, cols = struct.unpack('>IIII', header)
```

* `header = f_img.read(16)`：

  * 把 `f_img.read(16)` 的返回值赋给变量 `header`。
* 第二行是 **多重赋值 / 解包**：

  * `struct.unpack('>IIII', header)` 返回一个包含 4 个元素的元组，比如 `(2051, 60000, 28, 28)`。
  * `magic, num_images, rows, cols = ...` 会把这 4 个值按顺序“拆开放”到 4 个变量里。
  * 这种写法比：

    ```python
    t = struct.unpack(...)
    magic = t[0]
    num_images = t[1]
    ...
    ```

    要简洁很多。

类似的还有：

```python
magic, num_labels = struct.unpack('>II', header)
```

---

## 5. `if` 条件语句、比较运算符、`raise` 和 `ValueError`

```python
if magic != 2051:
    raise ValueError(f'Invalid magic number in image file: {magic}')
```

* `if 条件:`：条件为 True 时执行下面缩进的代码块。
* `!=`：不等于。
* `raise ValueError(...)`：

  * 主动抛出一个异常，类型是 `ValueError`（“值错误”）。
  * 这样如果文件格式不对，程序就不会默默出错，而是立刻报出清晰的错误信息。

### F-string（格式化字符串）

```python
f'Invalid magic number in image file: {magic}'
```

* 前面加 `f`，花括号 `{}` 中可以直接写变量或表达式。
* 会自动把变量值插入到字符串中。
* 比如 `magic = 1234` 时，这个字符串就是
  `'Invalid magic number in image file: 1234'`

---

## 6. 读取数据并用 `numpy` 处理

```python
img_data = f_img.read()
images = np.frombuffer(img_data, dtype=np.uint8)
```

* `f_img.read()`：

  * 不传参数时，读取**剩余的全部字节**。
* `np.frombuffer(img_data, dtype=np.uint8)`：

  * 语法：`numpy.frombuffer(buffer, dtype=...)`
  * 从一块连续的字节（buffer）里，按照指定 `dtype` 解释成一维数组。
  * 这里是 `uint8`，就是 0~255 的无符号 8 比特整数。

---

## 7. `reshape` 改变形状

```python
images = images.reshape(num_images, rows * cols)
```

* `reshape(新形状...)`：

  * 把数组重排成指定形状，但不改变元素总数。
  * 这里原始是 1 D 长度为 `num_images * rows * cols` 的数组，
  * 改成 2 D：`(num_images, rows * cols)`，每一行是一张图片铺平后的像素。

---

## 8. `astype` 转类型、除法运算

```python
X = images.astype(np.float32) / 255.0
```

* `images.astype(np.float32)`：

  * 把数组的数据类型从 `uint8` 转成 `float32`。
* `/ 255.0`：

  * 浮点除法，把 0~255 映射到 0.0~1.0。
  * 因为前面已经是 `float32`，结果仍然是 `float32` 类型。

---

## 9. 再次 `if` 检查 & 多行字符串参数

```python
if num_images != num_labels:
    raise ValueError(
        f'Image count ({num_images}) does not match label count ({num_labels})'
    )
```

* 条件同前。
* 这里 `raise ValueError(...)` 里面的参数跨两行写，是因为：

  * Python 允许在括号内部 **自动换行**。
  * 这样写方便排版，不是语法上的特殊含义。

---

## 10. `return` 返回多个值（元组）

```python
return X, y
```

* 函数里使用 `return` 返回结果。
* `return X, y` 实际上返回的是一个元组 `(X, y)`。
* 调用时一般写：

  ```python
  X, y = parse_mnist(..., ...)
  ```

  这也是一种解包：把返回的元组的两个元素分别赋给 `X` 和 `y`。

---

## 你可以做的小练习（加强语法理解）

你可以自己写个小脚本，不用 MNIST 文件，只练语法：

1. 练 **with 和 f-string**：

```python
with open('test.txt', 'w') as f:
    for i in range(3):
        f.write(f'line {i}\n')
```

2. 练 **多重赋值 / 解包**：

```python
t = (1, 2, 3)
a, b, c = t
print(a, b, c)  # 1 2 3
```

3. 练 **返回多个值**：

```python
def foo():
    return 10, 20

x, y = foo()
print(x, y)
```

---

好的，我们来详细讲讲 **`struct.unpack`** ——这是 Python 里解析二进制文件格式最核心的函数之一。

我会从直白概念开始讲，到格式字符串的含义，再到和 MNIST 的关系，最后配例子。

---

## 🚀 1. `struct.unpack` 是什么？

一句话：

> **`struct.unpack` 用来把二进制字节（bytes）转换成 Python 里的数字。**

因为二进制文件里存的不是字符，而是原始字节（0~255），需要按照特定格式解释。

例如，一段字节：

```
00 00 08 03
```

如果按照无符号 32 位整数（big-endian）解释，它就是：

```
2051
```

---

## 🚀 2. 语法

```python
struct.unpack(format, buffer)
```

* **format**：格式字符串（告诉 Python 如何解释这些字节）
* **buffer**：字节序列（bytes 或 bytearray）

返回值：**tuple（元组）**
因为可以一次解析多个数。

---

## 🚀 3. MNIST 中的例子

MNIST 的头部有 16 字节，格式是：

```
magic number  (4字节)
num images    (4字节)
rows          (4字节)
cols          (4字节)
```

都是：

* 4 字节
* 无符号整数
* 大端序（big endian）

我们是这样解析的：

```python
magic, num_images, rows, cols = struct.unpack('>IIII', header)
```

来解释 **`'>IIII'`**：

---

## 🚀 4. 格式字符串 format 的含义

格式字符串由两类信息组成：

---

### ◆ ① 字节序（endian）

开头的符号：

| 符号  | 含义                     |
| --- | ---------------------- |
| `>` | **大端序**（big-endian）    |
| `<` | 小端序（little-endian）     |
| `!` | network (= big-endian) |
| `=` | 本机默认字节序                |

MNIST 明确规定：**所有数都是 big-endian**
所以我们写 `>`。

---

### ◆ ② 数据类型

常用的：

| 字符  | 含义                  | 字节数     |
| --- | ------------------- | ------- |
| `I` | unsigned int（无符号整数） | 4 bytes |
| `H` | unsigned short      | 2 bytes |
| `B` | unsigned char       | 1 byte  |
| `f` | float               | 4 bytes |
| `d` | double              | 8 bytes |

MNIST 头里的每个数字都是 **4 字节无符号整数** → `I`

---

## ⭐ 所以解析 MNIST 头部用 `' >IIII '`：

* `>` → big-endian
* `I I I I` → 4 个 4 字节无符号整数

解析结果是一个 tuple：

```python
(2051, 60000, 28, 28)
```

---

## 🚀 5. 一个简单例子（帮助直观理解）

假设我们有 4 字节：

```python
data = b'\x00\x00\x08\x03'
```

我们解析：

```python
import struct
print(struct.unpack('>I', data))
```

输出：

```
(2051,)
```

---

#### 如果换成小端序：

```python
struct.unpack('<I', data)
```

解释成：

```
(50462976,)   # 完全不一样！字节序决定意义
```

---

## 🚀 6. 一次解析多个数字

假设我们有：

```python
data = b'\x00\x00\x08\x03\x00\x00\x00\x02'
```

可以一次解析两个整数：

```python
struct.unpack('>II', data)
```

输出：

```
(2051, 2)
```

---

## 🚀 7. 为什么 `unpack` 返回元组？

因为可能解析多个数据，例如 `'III'` 是三个整数。

在 MNIST 里：

```python
magic, num_images, rows, cols = struct.unpack('>IIII', header)
```

刚好一一对应。

---

## 🚀 8. 如果 buffer 长度不够，会怎么样？

例如：

```python
struct.unpack('>I', b'\x00\x01')
```

会报错：

```
struct.error: unpack requires a buffer of 4 bytes
```

因为你要求解析 4 字节，但只给了 2 字节 → 数据不完整。

---

## 🎉 总结（重点）

## `struct.unpack` 的核心作用：

✔ 把“二进制字节”按照指定格式解释成数字
✔ 常用于解析文件格式、协议、网络数据包

### MNIST 中为什么用 `'>IIII'`？

* `>`：大端序
* `I I I I`：四个 4 字节整数
* 刚好对应：

  * magic number
  * 图像数量
  * 行数
  * 列数

# `softmax_loss` 函数


``` python
def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.uint8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    rows = np.arange(Z.shape[0])
    log = np.log(np.sum(np.exp(Z), axis=1))
    return -np.mean(Z[rows, y] - log)
    ### END YOUR CODE

```

讲道理现在 deepseek 对算法的讲解确实挺不错的

好的，我来把 softmax loss 的核心公式整理到你的笔记中。

---

## Softmax Loss 核心公式

### 1. Softmax 函数
将 logits（原始预测分数）转换为概率分布：

$$ \text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}} $$

其中：
- $z_i$ 是第 i 个类别的 logit
- $C$ 是类别总数
- 输出是概率值，所有类别概率和为 1

### 2. Cross-Entropy Loss（交叉熵损失）
对于分类问题，使用交叉熵损失：

$ L = -\log(p_y) $

其中 $p_y$ 是真实类别 y 的预测概率。

### 3. Softmax + Cross-Entropy 组合
将两者结合，得到 softmax loss（交叉熵损失）：

$$ L = -\log\left(\frac{e^{z_y}}{\sum_{j=1}^{C} e^{z_j}}\right) = -z_y + \log\left(\sum_{j=1}^{C} e^{z_j}\right) $$

### 4. 批量计算的向量化形式
对于 batch_size = N 的批量数据：

$$ L = -\frac{1}{N}\sum_{i=1}^{N}\left[z_{i,y_i} - \log\left(\sum_{j=1}^{C} e^{z_{i,j}}\right)\right] $$

其中：
- $z_{i,y_i}$：第 i 个样本真实类别的 logit
- $\sum_{j=1}^{C} e^{z_{i,j}}$：第 i 个样本所有类别的 exp 求和

### 5. 代码实现对应关系
```python
def softmax_loss(Z, y):
    rows = np.arange(Z.shape[0])                    # 样本索引 [0, 1, ..., N-1]
    log = np.log(np.sum(np.exp(Z), axis=1))         # log(∑exp(z)) 部分
    return -np.mean(Z[rows, y] - log)               # -mean(z_y - log(∑exp(z)))
```

### 6. 梯度公式（反向传播）
softmax loss 对 logits 的梯度为：

$$ \frac{\partial L}{\partial z_i} = 
\begin{cases} 
p_i - 1 & \text{if } i = y \\
p_i & \text{if } i \neq y 
\end{cases} $$

其中 $p_i = \text{softmax}(z_i)$

### 7. 数值稳定版本
实际实现中需要考虑数值稳定性：

$$ L = -\frac{1}{N}\sum_{i=1}^{N}\left[(z_{i,y_i} - m_i) - \log\left(\sum_{j=1}^{C} e^{z_{i,j} - m_i}\right)\right] $$

其中 $m_i = \max_j z_{i,j}$，减去最大值避免 exp 溢出。

---

这些公式是理解 softmax loss 的核心，涵盖了从数学原理到代码实现的完整链条。

#  `softmax_regession_epoch`
``` python
def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data. """

    num_examples = X.shape[0]


    # 按顺序遍历所有 minibatch，不打乱
    for i in range(0, num_examples, batch):
        X_batch = X[i:i+batch]   # (B, D)
        y_batch = y[i:i+batch]   # (B,)
        B = X_batch.shape[0]     # 实际 batch 大小（最后一个 batch 可能不满）

        # 1. 计算 logits：Z = X_batch @ theta
        logits = X_batch @ theta    # 形状 (B, C)

        # 2. 数值稳定的 softmax,使用keepdims = True，保持维度方便广播
        logits -= np.max(logits, axis=1, keepdims=True)  # 每行减去最大值
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)  # (B, C)

        # 3. 构造 P - Y_onehot（直接在 probs 上操作）
        rows = np.arange(B)
        probs[rows, y_batch] -= 1.0   # 现在 probs = P - Y_onehot

        # 4. 计算梯度并更新 theta（使用平均 loss 的梯度）
        grad = X_batch.T @ probs / B   # (D, B) @ (B, C) -> (D, C)
        theta -= lr * grad             # in-place 更新

    # 题目要求：函数不返回任何值，theta 已在原地更新
    return None

```



``` python
def softmax_regression_epoch(X, y, theta, lr = 0.1, batch = 100):
	"""Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
	
	"""
	num_examples = X.shape[0]

	for i in range(0, num_examples, batch):
		X_batch = X[i:i+batch]
		y_batch = y[i:i+batch]
		B = X_batch.shape[0]
		
		logits = X_batch @ theta # (B,I) *(I,C) -> (B,C)
		# 数值稳定的softmax,keepdims保持维度（B,1）方便广播相减
		logits -= np.max(logits, axis=1, keepdims=True)
		exp_logits = np.exp(logits)
		probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
		
		## 构造Z - E_y
		## [rows,y]技巧
		rows = np.arange(B)
		probs[rows, y_batch] -= 1.0
		# 计算梯度
		grad = X_batch.T @ prob /B
		theta -= lr*grad
	return None
```



## 索引和越界问题
``` python
for i in range(0, num_examples, batch):
	# 如果i+batch超过了数组的长度，python会自动截取到数组末尾
	X_batch = X[i:i+batch]
	y_batch = y[i:i+batch]
	B = X_batch.shape[0]
```
有一个问题是就是 num_examples 不一定是 batch 的倍数。
但 python 中使用 `切片` 操作会自动截取到数组末尾，所以需要特意 `B = X_batch.shape[0]` 获取实际批大小，这样才能正确的构造 `Z - E_y`


# Softmax regression in C++

使用了 `pybind` 构建 C++和 python 的接口。

使用 C++来实现 `softmax regression`

要注意的是 C++就得手动来计算剩余样本了

- 循环条件：使用 current_batch 而不是 batch
- 索引计算：确保所有索引在有效范围内
- 内存访问：只访问已分配的内存区域
- 数组清零：在每次迭代前清零临时数组
