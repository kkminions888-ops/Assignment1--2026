# Assignment 1 错误修复说明

本文档记录了本次调试中修改过的所有代码位置。每一条都包含以下三部分内容：

1. 错误所在的文件与行号
2. 错误原因，以及它对训练或评估行为造成的影响
3. 具体是如何修改代码的

## 1. Embedding 与输入处理部分

### 1.1 Highway 中的转置维度错误

- 文件与行号：`Models/embedding.py:19`
- 错误原因：
  `Highway.forward()` 中原本使用的是 `x.transpose(0, 2)`，这会把 batch 维和序列维交换。对于输入张量 `[B, C, L]`，模型真正需要的是转成 `[B, L, C]`，而不是把 batch 维打乱。
- 对训练/评估的影响：
  这会导致后续线性层读取到错误的张量结构，embedding 模块从一开始就输出错误结果，后面的编码层也会全部受到影响，训练很难正常进行。
- 修改方式：
  将 `x.transpose(0, 2)` 改为 `x.transpose(1, 2)`，正确地把 `[B, C, L]` 变为 `[B, L, C]`。

### 1.2 字符 embedding 的维度重排错误

- 文件与行号：`Models/embedding.py:39`
- 错误原因：
  原代码使用 `permute(0, 2, 1, 3)`，会把第二维变成序列长度，而不是字符 embedding 通道数。
- 对训练/评估的影响：
  字符卷积层期望的输入是 `[B, d_char, L, char_len]`。如果维度顺序不对，卷积层拿到的“通道数”就错了，轻则特征含义完全错误，重则直接报 shape 错误。
- 修改方式：
  将 `permute(0, 2, 1, 3)` 改为 `permute(0, 3, 1, 2)`，使字符 embedding 进入卷积层前的形状符合预期。

### 1.3 词向量和字符向量的 embedding 调用写反了

- 文件与行号：`Models/qanet.py:65`
- 错误原因：
  原代码把 `self.char_emb` 用在了词 id `Cwid` 上，把 `self.word_emb` 用在了字符 id `Ccid` 上。
- 对训练/评估的影响：
  词表索引和字符表索引本来就来自不同的词典，语义和索引范围都不一样。写反之后会直接污染输入表示，甚至可能出现 embedding lookup 越界。
- 修改方式：
  改成 `self.word_emb(Cwid)` 和 `self.char_emb(Ccid)`，让词 id 走词 embedding，字符 id 走字符 embedding。

### 1.4 Context-Question Attention 的 mask 参数顺序颠倒

- 文件与行号：`Models/qanet.py:75`
- 错误原因：
  原代码调用 `self.cq_att(Ce, Qe, qmask, cmask)`，把 question 的 mask 和 context 的 mask 传反了。
- 对训练/评估的影响：
  attention 中 padding 位会被错误屏蔽，导致模型把本该忽略的位置当作有效信息参与计算，训练和评估的结果都会变差。
- 修改方式：
  改成 `self.cq_att(Ce, Qe, cmask, qmask)`。

## 2. 卷积、注意力与编码器模块

### 2.1 自定义 Conv1d 展开的维度错了

- 文件与行号：`Models/conv.py:55`
- 错误原因：
  原代码使用 `x.unfold(1, ...)`，实际上是在通道维上滑窗，而不是在序列长度维上滑窗。
- 对训练/评估的影响：
  这会让 1D 卷积的数学意义完全错误，模型中的所有卷积编码都会受到影响。
- 修改方式：
  改成 `x.unfold(2, self.kernel_size, 1)`，在长度维上做滑动窗口。

### 2.2 自定义 Conv2d 的宽度 padding 使用了旧高度

- 文件与行号：`Models/conv.py:124`
- 错误原因：
  在高度 padding 完成后，原代码仍然用旧的 `H` 去创建宽度方向的 padding。
- 对训练/评估的影响：
  这会让 padding 张量和当前输入张量的高度不匹配，从而破坏 2D 卷积，尤其影响字符卷积分支。
- 修改方式：
  改成使用 `x.size(2)`，保证宽度 padding 的高度与当前张量一致。

### 2.3 Depthwise Separable Convolution 的执行顺序反了

- 文件与行号：`Models/conv.py:175`
- 错误原因：
  原代码先做 pointwise convolution，再做 depthwise convolution。
- 对训练/评估的影响：
  这不符合 depthwise separable convolution 的标准结构，模型实际结构会偏离设计目标。
- 修改方式：
  改成 `self.pointwise_conv(self.depthwise_conv(x))`，先 depthwise，再 pointwise。

### 2.4 Context-Question Attention 中矩阵乘法顺序错误

- 文件与行号：`Models/attention.py:38`
- 错误原因：
  原代码写的是 `A = torch.bmm(Q, S1)`。
- 对训练/评估的影响：
  这不符合 context-to-question attention 的标准计算方式，张量语义也不正确，会导致 attention 输出失真。
- 修改方式：
  改成 `A = torch.bmm(S1, Q)`。

### 2.5 Multi-Head Attention 中 head 维和 batch 维重排错误

- 文件与行号：`Models/encoder.py:70`
- 错误原因：
  原代码使用 `permute(2, 0, 1, 3)`，把 head 维放到了 batch 维前面，打乱了预期的张量布局。
- 对训练/评估的影响：
  注意力计算时 batch 和 head 的对应关系不正确，最终 attention 结果会错误。
- 修改方式：
  将 `q`、`k`、`v` 的重排统一改成 `permute(0, 2, 1, 3)`。

### 2.6 Multi-Head Attention 缺少缩放因子

- 文件与行号：`Models/encoder.py:78`
- 错误原因：
  点积注意力没有乘上 `1 / sqrt(d_k)`。
- 对训练/评估的影响：
  注意力分数可能过大，softmax 容易饱和，训练会变得不稳定。
- 修改方式：
  在 attention score 上乘以 `self.scale`。

### 2.7 Multi-Head Attention 合并各个 head 时维度顺序错误

- 文件与行号：`Models/encoder.py:85`
- 错误原因：
  原代码用了 `permute(1, 2, 0, 3)` 来把 head 合并回去，这会再次打乱 batch 维和 head 维。
- 对训练/评估的影响：
  即使前面 attention 算对了，最后输出恢复时也会产生错误结构。
- 修改方式：
  改成 `permute(0, 2, 1, 3)`，再 reshape 回 `[B, L, d_model]`。

### 2.8 EncoderBlock 中 normalization 下标使用错误

- 文件与行号：`Models/encoder.py:121`
- 错误原因：
  原代码在循环里使用 `self.norms[i + 1]`。
- 对训练/评估的影响：
  这会导致 normalization 层索引偏移，最后一层甚至可能直接越界。
- 修改方式：
  改为 `self.norms[i]`，使每个卷积块对应自己的 normalization 层。

### 2.9 EncoderBlock 丢掉了 self-attention 的输出

- 文件与行号：`Models/encoder.py:117`
- 错误原因：
  原代码在执行完 self-attention 后，直接 `out = res`，把 attention 的输出覆盖掉了。
- 对训练/评估的影响：
  这样等于 self-attention 根本没有真正参与模型表示学习，QANet 的核心能力会明显受损。
- 修改方式：
  改成残差连接 `out = out + res`，保留 self-attention 输出。

### 2.10 Pointer Head 在错误的维度上拼接

- 文件与行号：`Models/heads.py:23`
- 错误原因：
  原代码写成了 `torch.cat([M1, M2], dim=0)`，沿 batch 维进行拼接。
- 对训练/评估的影响：
  指针网络本来应该拼接通道维来生成起始位置和结束位置的 logits，沿 batch 维拼接会直接破坏输入结构。
- 修改方式：
  改成 `dim=1`，沿通道维拼接。

## 3. 激活函数、Dropout、Normalization 与初始化

### 3.1 Inverted Dropout 缩放因子写错

- 文件与行号：`Models/dropout.py:17`
- 错误原因：
  原代码除的是 `self.p`，而标准 inverted dropout 应该除以保留概率 `1 - p`。
- 对训练/评估的影响：
  surviving activation 的期望值会错误，导致训练时数值分布被放大或缩小，训练不稳定。
- 修改方式：
  改成 `return x * mask / (1.0 - self.p)`。

### 3.2 ReLU 实现方向反了

- 文件与行号：`Models/Activations/relu.py:12`
- 错误原因：
  原代码使用 `x.clamp(max=0.0)`，相当于保留负数、把正数截成 0。
- 对训练/评估的影响：
  这与 ReLU 的定义完全相反，会破坏整个网络中的非线性表示能力。
- 修改方式：
  改成 `x.clamp(min=0.0)`。

### 3.3 LeakyReLU 实现方向反了

- 文件与行号：`Models/Activations/leakeyReLU.py:19`
- 错误原因：
  原代码对负数不做缩放，反而对正数乘了 `negative_slope`。
- 对训练/评估的影响：
  这会使正激活被错误压缩，模型非线性行为和预期不一致。
- 修改方式：
  改成 `torch.where(x < 0, self.negative_slope * x, x)`。

### 3.4 LayerNorm 的统计量广播和仿射公式都错了

- 文件与行号：`Models/Normalizations/layernorm.py:37`, `Models/Normalizations/layernorm.py:41`
- 错误原因：
  原代码在计算 mean 和 variance 时使用了 `keepdim=False`，不方便和原张量广播对齐；同时仿射变换写成了 `x_norm * bias + weight`，顺序也不对。
- 对训练/评估的影响：
  可能出现 shape 不匹配，也可能即使能运行，归一化结果也是错误的。
- 修改方式：
  将 mean 和 variance 改为 `keepdim=True`，并把输出公式改为 `x_norm * self.weight + self.bias`。

### 3.5 GroupNorm 分组方式错误

- 文件与行号：`Models/Normalizations/groupnorm.py:35`
- 错误原因：
  原代码 reshape 成了 `[B, C // G, G, ...]`，正确方式应当是 `[B, G, C // G, ...]`。
- 对训练/评估的影响：
  通道分组被打乱，归一化统计量不再是按正确 group 计算的。
- 修改方式：
  改成 `x.view(B, self.G, C // self.G, *spatial)`。

### 3.6 Kaiming 初始化方差公式错误

- 文件与行号：`Models/Initializations/kaiming.py:25`
- 错误原因：
  原代码使用的是 `sqrt(1 / fan)`，而 Kaiming 初始化应为 `sqrt(2 / fan)`。
- 对训练/评估的影响：
  权重初始化过小，会降低 ReLU 网络的训练效率。
- 修改方式：
  将 Kaiming normal 和 Kaiming uniform 中的公式都改为 `sqrt(2.0 / fan)`。

### 3.7 Xavier 初始化公式错误

- 文件与行号：`Models/Initializations/xavier.py:24`
- 错误原因：
  原代码使用的是 `fan_in * fan_out`，但 Xavier 初始化应使用 `fan_in + fan_out`。
- 对训练/评估的影响：
  初始化方差会过小，网络训练更困难。
- 修改方式：
  将 Xavier normal 和 Xavier uniform 的公式都改成使用 `fan_in + fan_out`。

## 4. Loss、优化器与学习率调度器

### 4.1 NLL Loss 参数顺序写反

- 文件与行号：`Losses/loss.py:7`
- 错误原因：
  原代码调用 `F.nll_loss(y1, p1)`，但 PyTorch 的参数顺序应该是 `(input, target)`。
- 对训练/评估的影响：
  训练会在计算 loss 时直接报错，因为输入张量和标签张量位置错了。
- 修改方式：
  改成 `F.nll_loss(p1, y1)`，另一项保持为 `F.nll_loss(p2, y2)`。

### 4.2 Adam 工厂函数忽略了 `learning_rate`

- 文件与行号：`Optimizers/optimizer.py:14`
- 错误原因：
  原代码把 Adam 的学习率硬编码为 `1.0`。
- 对训练/评估的影响：
  训练配置里设置的学习率会失效，导致模型以错误的步长更新参数。
- 修改方式：
  改成 `lr=args.learning_rate`。

### 4.3 Adam 的 weight decay 符号错误

- 文件与行号：`Optimizers/adam.py:53`
- 错误原因：
  原代码使用 `alpha=-wd`，相当于把 weight decay 的方向写反了。
- 对训练/评估的影响：
  原本应当把参数往 0 拉回的正则项，反而可能把参数往相反方向推。
- 修改方式：
  改成 `alpha=wd`。

### 4.4 Adam 读取了错误的 state key

- 文件与行号：`Optimizers/adam.py:63`
- 错误原因：
  初始化时写入的是 `exp_avg` 和 `exp_avg_sq`，但后面却读取 `state["m"]` 和 `state["v"]`。
- 对训练/评估的影响：
  Adam 在第一次更新时就会因为 key 不存在而报错。
- 修改方式：
  改成读取 `state["exp_avg"]` 和 `state["exp_avg_sq"]`。

### 4.5 Adam 二阶矩更新公式错误

- 文件与行号：`Optimizers/adam.py:69`
- 错误原因：
  原代码把原始梯度直接加进二阶矩累计，而不是累加梯度平方。
- 对训练/评估的影响：
  这样实现的就不是 Adam，会导致优化过程数值不正确。
- 修改方式：
  改成 `addcmul_(grad, grad, value=1.0 - beta2)`。

### 4.6 Adam 的 bias correction 公式错误

- 文件与行号：`Optimizers/adam.py:72`
- 错误原因：
  原代码写成了 `1 - beta * t`，正确应为 `1 - beta ** t`。
- 对训练/评估的影响：
  bias correction 数值会严重错误，尤其在训练早期影响更明显。
- 修改方式：
  改成幂运算版本的标准公式。

### 4.7 SGD 的 weight decay 符号错误

- 文件与行号：`Optimizers/sgd.py:39`
- 错误原因：
  原代码同样使用了 `alpha=-wd`。
- 对训练/评估的影响：
  weight decay 方向反了，起不到正常 L2 正则化的作用。
- 修改方式：
  改为 `alpha=wd`。

### 4.8 SGD Momentum 的 velocity 键名不一致

- 文件与行号：`Optimizers/sgd_momentum.py:49`
- 错误原因：
  原代码创建的是 `state["vel"]`，但读取时却使用 `state["velocity"]`。
- 对训练/评估的影响：
  使用 momentum 优化器时会直接因为取不到 state 而报错。
- 修改方式：
  统一改为 `state["velocity"]`。

### 4.9 SGD Momentum 的速度更新方向错误

- 文件与行号：`Optimizers/sgd_momentum.py:54`
- 错误原因：
  原代码使用 `v.mul_(mu).sub_(grad)`。
- 对训练/评估的影响：
  这与标准 SGD with Momentum 的更新方向不一致，可能导致优化行为失真。
- 修改方式：
  改为 `v.mul_(mu).add_(grad)`。

### 4.10 Lambda Scheduler 用了加法而不是乘法

- 文件与行号：`Schedulers/lambda_scheduler.py:23`
- 错误原因：
  原代码返回的是 `base_lr + factor`。
- 对训练/评估的影响：
  学习率调度逻辑会完全偏离预期，因为 LambdaLR 应该是对基础学习率做乘法缩放。
- 修改方式：
  改成 `base_lr * factor`。

### 4.11 Step Scheduler 公式错误

- 文件与行号：`Schedulers/step_scheduler.py:25`
- 错误原因：
  原代码计算的是 `base_lr * gamma * floor(t / step_size)`。
- 对训练/评估的影响：
  这会让 step 0 时学习率就可能变成 0，也不符合标准 step decay 行为。
- 修改方式：
  改成 `base_lr * (gamma ** (t // step_size))`。

### 4.12 Cosine Scheduler 公式不完整且使用了错误的 pi 常量

- 文件与行号：`Schedulers/cosine_scheduler.py:28`
- 错误原因：
  原代码少了 `0.5` 系数，而且写成了 `math.PI`，但 Python 中正确的是 `math.pi`。
- 对训练/评估的影响：
  要么直接报错，要么学习率曲线错误。
- 修改方式：
  改成标准 cosine annealing 公式，并使用 `math.pi`。

### 4.13 notebook 默认使用的 `"none"` scheduler 没有注册

- 文件与行号：`Schedulers/scheduler.py:28`, `Schedulers/scheduler.py:37`
- 错误原因：
  notebook 训练代码里使用 `scheduler_name="none"`，但 scheduler registry 里原本没有这个键。
- 对训练/评估的影响：
  训练一开始就会在 scheduler 查找阶段报 `ValueError`。
- 修改方式：
  新增 `none_scheduler()` 作为 no-op scheduler，并在 registry 中注册 `"none"`。

## 5. 训练与评估流程

### 5.1 `argparse.Namespace` 的构造方式错误

- 文件与行号：`TrainTools/train.py:107`
- 错误原因：
  原代码写成了 `argparse.Namespace({k: v ...})`，把字典作为位置参数传入。
- 对训练/评估的影响：
  后续代码希望通过 `args.xxx` 的方式访问参数，但这个 `Namespace` 构造方式本身就是错的，会导致训练流程出问题。
- 修改方式：
  改成 `argparse.Namespace(**{k: v for k, v in locals().items()})`。

### 5.2 反向传播错误地调用在 `loss.item()` 上

- 文件与行号：`TrainTools/train_utils.py:34`
- 错误原因：
  `loss.item()` 会把张量转成 Python float，原代码随后对这个 float 调用了 `.backward()`。
- 对训练/评估的影响：
  训练第一步就会报错，因为 float 没有梯度信息。
- 修改方式：
  改成直接对张量调用 `loss.backward()`。

### 5.3 梯度裁剪顺序错误

- 文件与行号：`TrainTools/train_utils.py:35-36`
- 错误原因：
  原代码先 `optimizer.step()`，再做 `clip_grad_norm_()`。
- 对训练/评估的影响：
  梯度裁剪对已经完成的参数更新没有任何作用，相当于没裁剪。
- 修改方式：
  把顺序改成先 gradient clipping，再 `optimizer.step()`。

### 5.4 评估阶段读取 checkpoint 的 key 不匹配

- 文件与行号：`EvaluateTools/evaluate.py:119`
- 错误原因：
  训练保存 checkpoint 时使用的是 `model_state`，而评估代码原本读取的是 `model`。
- 对训练/评估的影响：
  即使训练结束了，评估阶段也会因为找不到正确的模型参数 key 而失败。
- 修改方式：
  改成优先读取 `model_state`，同时兼容旧的 `model` 键。

### 5.5 评估时 `argmax` 取错了维度

- 文件与行号：`EvaluateTools/eval_utils.py:107-108`
- 错误原因：
  原代码对形状为 `[B, L]` 的输出使用了 `dim=0`。
- 对训练/评估的影响：
  这样取得的是“整个 batch 中最优的位置”，而不是“每个样本自己的最优位置”，会导致预测 span 全部错误，F1 和 EM 失真。
- 修改方式：
  将两个 `argmax` 都改成 `dim=1`，让每个样本独立预测自己的起止位置。

## 总结

本次修复主要覆盖了以下几类问题：

- QANet 输入表示和 mask 传递错误
- 自定义卷积实现错误
- 注意力与 pointer head 的张量维度错误
- 激活函数、dropout、normalization、初始化公式错误
- 优化器与学习率调度器实现错误
- 训练/评估流程中的 backward、checkpoint 读取、预测解码错误

这些修改完成后，项目源码已经能够通过静态编译检查，主 notebook 流程也比原始版本更接近可在 Colab 中正常运行的状态。
