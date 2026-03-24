# Assignment 1 Bug Fix Report

This document records every code change made during debugging. For each issue, it states:

1. the file and line number where the bug was fixed
2. why the original code was wrong and how it affected training or evaluation
3. how the code was changed

## 1. Embedding and Input Pipeline

### 1.1 Highway transpose bug

- File and line: `Models/embedding.py:19`
- Original problem:
  The `Highway.forward()` function used `x.transpose(0, 2)`, which swaps the batch dimension with the sequence dimension. For an input shaped `[B, C, L]`, this produces an invalid layout for the following linear layers, because the model expects `[B, L, C]`.
- Impact:
  This breaks the embedding projection path and causes incorrect tensor shapes early in the model. Training cannot proceed correctly because later layers receive malformed activations.
- Fix:
  Changed the transpose to `x.transpose(1, 2)`, which correctly converts `[B, C, L]` into `[B, L, C]`.

### 1.2 Character embedding permutation bug

- File and line: `Models/embedding.py:39`
- Original problem:
  The character embedding tensor was permuted with `permute(0, 2, 1, 3)`. That makes the second dimension equal to sequence length instead of character embedding channels.
- Impact:
  The 2D convolution in the character encoder expects the input shape `[B, d_char, L, char_len]`. With the wrong permutation, the convolution sees the wrong number of input channels and will fail or learn nonsense.
- Fix:
  Changed the permutation to `permute(0, 3, 1, 2)`, so the tensor layout becomes `[B, d_char, L, char_len]`.

### 1.3 Word and character embeddings were swapped

- File and line: `Models/qanet.py:65`
- Original problem:
  The code applied `self.char_emb` to word indices and `self.word_emb` to character indices for the context branch.
- Impact:
  Word IDs and character IDs come from different vocabularies and have different semantic meanings and tensor shapes. Swapping them corrupts the model input and can also trigger out-of-range embedding lookups.
- Fix:
  Replaced the assignments so that `self.word_emb` is applied to `Cwid` and `self.char_emb` is applied to `Ccid`.

### 1.4 Context-question attention mask order was reversed

- File and line: `Models/qanet.py:75`
- Original problem:
  `self.cq_att(Ce, Qe, qmask, cmask)` passed the question mask where the context mask should go, and vice versa.
- Impact:
  Padding positions are masked incorrectly during context-question attention. This distorts attention weights and degrades both training and evaluation quality.
- Fix:
  Changed the call to `self.cq_att(Ce, Qe, cmask, qmask)`.

## 2. Convolution, Attention, and Encoder Blocks

### 2.1 Custom Conv1d unfolded the wrong dimension

- File and line: `Models/conv.py:55`
- Original problem:
  The custom 1D convolution used `x.unfold(1, ...)`, which unfolds over the channel dimension instead of the sequence-length dimension.
- Impact:
  The extracted sliding windows are wrong, so the convolution is mathematically incorrect. This affects every encoder convolution in QANet.
- Fix:
  Changed the unfold call to `x.unfold(2, self.kernel_size, 1)`, which correctly slides over the length axis.

### 2.2 Custom Conv2d width padding used the old height

- File and line: `Models/conv.py:124`
- Original problem:
  After height padding, the code still used the old `H` value when creating width padding.
- Impact:
  The padding tensor can have incompatible dimensions after the height has already changed, which breaks the custom 2D convolution path used in character embedding.
- Fix:
  Replaced `H` with `x.size(2)` so width padding matches the current padded height.

### 2.3 Depthwise separable convolution order was reversed

- File and line: `Models/conv.py:175`
- Original problem:
  The code applied pointwise convolution first and depthwise convolution second.
- Impact:
  That is not a proper depthwise separable convolution. It changes both the intended architecture and the tensor semantics of later layers.
- Fix:
  Changed the forward pass to `self.pointwise_conv(self.depthwise_conv(x))`, which performs depthwise first and pointwise second.

### 2.4 Context-question attention matrix multiplication used the wrong order

- File and line: `Models/attention.py:38`
- Original problem:
  The code computed `A = torch.bmm(Q, S1)`.
- Impact:
  The tensor dimensions do not correspond to the standard QANet context-to-question attention computation. The attention output should be produced by multiplying the attention weights by the question representation, not the other way around.
- Fix:
  Changed the line to `A = torch.bmm(S1, Q)`.

### 2.5 Multi-head attention reshaped heads incorrectly

- File and line: `Models/encoder.py:70`
- Original problem:
  The query, key, and value tensors were permuted with `permute(2, 0, 1, 3)`, which puts head index before batch index.
- Impact:
  This mixes batch and head dimensions incorrectly and produces invalid attention batches.
- Fix:
  Changed the permutation to `permute(0, 2, 1, 3)` for `q`, `k`, and `v`, so the layout becomes `[B, heads, L, d_k]` before flattening.

### 2.6 Multi-head attention missed the scale factor

- File and line: `Models/encoder.py:78`
- Original problem:
  Dot-product attention was computed without multiplying by `1 / sqrt(d_k)`.
- Impact:
  The logits can become too large, pushing softmax into saturation and making optimization unstable.
- Fix:
  Multiplied the attention scores by `self.scale`.

### 2.7 Multi-head attention merged heads back incorrectly

- File and line: `Models/encoder.py:85`
- Original problem:
  The output tensor was permuted with `permute(1, 2, 0, 3)` before reshaping.
- Impact:
  This again mixes batch and head dimensions and reconstructs the output in the wrong order.
- Fix:
  Changed it to `permute(0, 2, 1, 3)` before reshaping back to `[B, L, d_model]`.

### 2.8 Encoder block indexed normalization layers incorrectly

- File and line: `Models/encoder.py:121`
- Original problem:
  The convolution loop used `self.norms[i + 1]`.
- Impact:
  On the last iteration this can exceed the available indices. Even before that, every convolution block uses the wrong normalization layer.
- Fix:
  Changed the access to `self.norms[i]`.

### 2.9 Encoder block dropped the self-attention result

- File and line: `Models/encoder.py:117`
- Original problem:
  After computing self-attention, the code overwrote the output with `res`.
- Impact:
  The self-attention module effectively has no effect, which severely damages QANet’s modeling capacity.
- Fix:
  Changed the residual connection to `out = out + res`, preserving the self-attention output.

### 2.10 Pointer head concatenated along the batch dimension

- File and line: `Models/heads.py:23`
- Original problem:
  `torch.cat([M1, M2], dim=0)` concatenated tensors across the batch dimension instead of the channel dimension.
- Impact:
  The pointer head receives a malformed representation and cannot produce valid start logits.
- Fix:
  Changed the concatenation axis to `dim=1`.

## 3. Activation, Dropout, Normalization, and Initialization

### 3.1 Inverted dropout used the wrong scaling factor

- File and line: `Models/dropout.py:17`
- Original problem:
  The code divided by `self.p` instead of `1 - self.p`.
- Impact:
  Surviving activations are scaled incorrectly, so the expected activation magnitude is wrong during training. This destabilizes optimization.
- Fix:
  Changed the scaling to `1.0 - self.p`.

### 3.2 ReLU implementation was reversed

- File and line: `Models/Activations/relu.py:12`
- Original problem:
  The code used `x.clamp(max=0.0)`, which keeps only negative values and clips positive values to zero.
- Impact:
  This is the opposite of ReLU and destroys the intended nonlinearity used throughout the model.
- Fix:
  Changed it to `x.clamp(min=0.0)`.

### 3.3 LeakyReLU implementation was reversed

- File and line: `Models/Activations/leakeyReLU.py:19`
- Original problem:
  Negative values were left unchanged and positive values were multiplied by `negative_slope`.
- Impact:
  This reverses the definition of LeakyReLU and weakens useful activations instead of preserving them.
- Fix:
  Changed the expression to `torch.where(x < 0, self.negative_slope * x, x)`.

### 3.4 LayerNorm used non-broadcastable statistics and affine terms in the wrong order

- File and lines: `Models/Normalizations/layernorm.py:37`, `Models/Normalizations/layernorm.py:41`
- Original problem:
  The mean and variance were computed with `keepdim=False`, which breaks broadcasting back to the original tensor shape. The affine transform also used `x_norm * bias + weight`, which is the reverse of standard LayerNorm.
- Impact:
  Normalization can fail due to shape mismatch, and even if it runs, the output is numerically wrong.
- Fix:
  Computed mean and variance with `keepdim=True`, and changed the affine transform to `x_norm * self.weight + self.bias`.

### 3.5 GroupNorm grouped channels with the wrong layout

- File and line: `Models/Normalizations/groupnorm.py:35`
- Original problem:
  The tensor was reshaped as `[B, C // G, G, ...]` instead of `[B, G, C // G, ...]`.
- Impact:
  Channels are assigned to groups incorrectly, so normalization statistics are computed over the wrong elements.
- Fix:
  Changed the reshape to `x.view(B, self.G, C // self.G, *spatial)`.

### 3.6 Kaiming initialization used the wrong variance

- File and line: `Models/Initializations/kaiming.py:25`
- Original problem:
  The standard deviation used `sqrt(1 / fan)` instead of `sqrt(2 / fan)`.
- Impact:
  Weights are initialized too small for ReLU-based networks, which can slow or weaken learning.
- Fix:
  Changed the standard deviation formula to `sqrt(2.0 / fan)` in both Kaiming normal and Kaiming uniform.

### 3.7 Xavier initialization used `fan_in * fan_out` instead of `fan_in + fan_out`

- File and line: `Models/Initializations/xavier.py:24`
- Original problem:
  The denominator used the product of `fan_in` and `fan_out`, which is not the Xavier formula.
- Impact:
  The initialized variance becomes far too small, making optimization harder.
- Fix:
  Changed the formula to use `fan_in + fan_out` in both Xavier normal and Xavier uniform.

## 4. Loss, Optimizers, and Learning-Rate Schedulers

### 4.1 NLL loss arguments were reversed

- File and line: `Losses/loss.py:7`
- Original problem:
  The original code called `F.nll_loss(y1, p1)`, but PyTorch expects `(input, target)`, not `(target, input)`.
- Impact:
  Training fails immediately because the loss function receives tensors of the wrong meaning and shape.
- Fix:
  Changed the call to `F.nll_loss(p1, y1)` and similarly kept `F.nll_loss(p2, y2)`.

### 4.2 Adam factory ignored `learning_rate`

- File and line: `Optimizers/optimizer.py:14`
- Original problem:
  The Adam optimizer factory hard-coded `lr=1.0`.
- Impact:
  The effective learning rate no longer matches the training configuration. This is especially harmful when the scheduler is intended to be a no-op or when the notebook specifies a custom learning rate.
- Fix:
  Changed the Adam factory to pass `lr=args.learning_rate`.

### 4.3 Adam weight decay sign was wrong

- File and line: `Optimizers/adam.py:53`
- Original problem:
  Weight decay used `alpha=-wd`, which moves the gradient in the wrong direction.
- Impact:
  Instead of regularizing weights, the optimizer can push them away from zero.
- Fix:
  Changed the sign to `alpha=wd`.

### 4.4 Adam read the wrong state keys

- File and line: `Optimizers/adam.py:63`
- Original problem:
  The state dictionary initialized `exp_avg` and `exp_avg_sq`, but later tried to read `state["m"]` and `state["v"]`.
- Impact:
  Adam fails on the first training step with a key error.
- Fix:
  Changed the code to read `state["exp_avg"]` and `state["exp_avg_sq"]`.

### 4.5 Adam second moment update used raw gradients instead of squared gradients

- File and line: `Optimizers/adam.py:69`
- Original problem:
  The variance accumulator used `add_(grad, ...)` instead of adding `grad * grad`.
- Impact:
  Adam no longer matches the Adam algorithm, so its updates are mathematically incorrect and unstable.
- Fix:
  Replaced the update with `addcmul_(grad, grad, value=1.0 - beta2)`.

### 4.6 Adam bias correction formula was wrong

- File and line: `Optimizers/adam.py:72`
- Original problem:
  Bias correction used `1 - beta * t` instead of `1 - beta ** t`.
- Impact:
  The correction factors are numerically wrong, especially in early training steps.
- Fix:
  Changed the correction terms to exponentiation-based formulas.

### 4.7 SGD weight decay sign was wrong

- File and line: `Optimizers/sgd.py:39`
- Original problem:
  Weight decay again used `alpha=-wd`.
- Impact:
  The optimizer regularizes in the wrong direction.
- Fix:
  Changed it to `alpha=wd`.

### 4.8 SGD with momentum stored the velocity under one key and read another

- File and line: `Optimizers/sgd_momentum.py:49`
- Original problem:
  The code created `state["vel"]` but then read `state["velocity"]`.
- Impact:
  Training fails with a missing-key error when momentum is used.
- Fix:
  Standardized the state key to `state["velocity"]`.

### 4.9 SGD with momentum updated velocity with the wrong sign

- File and line: `Optimizers/sgd_momentum.py:54`
- Original problem:
  The update used `v.mul_(mu).sub_(grad)`.
- Impact:
  The momentum direction becomes inconsistent with standard SGD-momentum, which can invert or distort optimization behavior.
- Fix:
  Changed it to `v.mul_(mu).add_(grad)`.

### 4.10 Lambda scheduler added instead of multiplied

- File and line: `Schedulers/lambda_scheduler.py:23`
- Original problem:
  The scheduler returned `base_lr + factor` instead of `base_lr * factor`.
- Impact:
  The scheduler does not behave like a multiplicative learning-rate scheduler and produces incorrect learning rates.
- Fix:
  Changed the return expression to multiplication.

### 4.11 Step scheduler formula was wrong

- File and line: `Schedulers/step_scheduler.py:25`
- Original problem:
  The scheduler computed `base_lr * gamma * floor(t / step_size)`.
- Impact:
  This can set the learning rate to zero at step 0 and does not implement step decay correctly.
- Fix:
  Changed the formula to `base_lr * (gamma ** (t // step_size))`.

### 4.12 Cosine scheduler formula missed the 0.5 factor and used the wrong pi constant

- File and line: `Schedulers/cosine_scheduler.py:28`
- Original problem:
  The cosine schedule omitted the `0.5` factor and used `math.PI`, which does not exist in Python.
- Impact:
  The scheduler either raises an error or computes the wrong learning-rate range.
- Fix:
  Replaced the expression with `0.5 * (1 + math.cos(math.pi * t / self.T_max))` inside the standard cosine annealing formula.

### 4.13 Notebook default scheduler name was unsupported

- File and lines: `Schedulers/scheduler.py:28`, `Schedulers/scheduler.py:37`
- Original problem:
  The notebook calls training with `scheduler_name="none"`, but the scheduler registry only contained `cosine`, `step`, and `lambda`.
- Impact:
  Training fails before starting because the scheduler lookup raises `ValueError`.
- Fix:
  Added `none_scheduler()` as a no-op scheduler and registered it under the `"none"` key.

## 5. Training and Evaluation Pipeline

### 5.1 `argparse.Namespace` was constructed incorrectly

- File and line: `TrainTools/train.py:107`
- Original problem:
  The code used `argparse.Namespace({k: v ...})`, passing a dictionary as a positional argument.
- Impact:
  The training configuration object is not created correctly, so later code that expects attributes on `args` breaks.
- Fix:
  Changed it to `argparse.Namespace(**{k: v for k, v in locals().items()})`.

### 5.2 Backpropagation was called on `loss.item()`

- File and line: `TrainTools/train_utils.py:34`
- Original problem:
  `loss.item()` converts the tensor loss into a Python float, and the code then tried to call `.backward()` on that float.
- Impact:
  Training crashes on the first optimization step because Python floats do not have gradients.
- Fix:
  Changed the code to call `loss.backward()` directly on the tensor.

### 5.3 Gradient clipping happened after the optimizer step

- File and lines: `TrainTools/train_utils.py:35-36`
- Original problem:
  The code performed `optimizer.step()` before clipping gradients.
- Impact:
  Gradient clipping had no effect on the parameter update that had just occurred.
- Fix:
  Reordered the operations so clipping happens before `optimizer.step()`.

### 5.4 Evaluation checkpoint loading expected the wrong key

- File and line: `EvaluateTools/evaluate.py:119`
- Original problem:
  The evaluation code previously loaded `ckpt["model"]`, while the training code saves `model_state`.
- Impact:
  Evaluation fails even after training completes because the checkpoint contents do not match what the loader expects.
- Fix:
  Changed the code to read `model_state`, while also allowing a fallback to `model` for compatibility.

### 5.5 Evaluation chose answer spans along the wrong dimension

- File and lines: `EvaluateTools/eval_utils.py:107-108`
- Original problem:
  The code used `torch.argmax(..., dim=0)` on tensors shaped `[B, L]`.
- Impact:
  This selects the best position across the batch instead of within each example, producing invalid predicted spans and meaningless evaluation scores.
- Fix:
  Changed both calls to `dim=1`, so each sample predicts its own best start and end index.

## Summary

The repaired issues cover:

- tensor layout and mask handling in QANet
- custom convolution correctness
- attention and pointer computations
- activation, dropout, normalization, and initialization math
- optimizer and scheduler correctness
- training/evaluation pipeline bugs such as backward, checkpoint loading, and evaluation span decoding

After these fixes, the Python source files pass static compilation checks, and the main notebook pipeline is much closer to running correctly in Colab.
