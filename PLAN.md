# CortexCPP Plan

CortexCPP is a deep learning framework written from scratch in C++. It can already describe, validate, and summarize CNN and MLP architectures. It cannot train them yet.

`PLAN.txt` is the original stage list. This file is the living plan: what the sources actually do, what is still on that roadmap, and what is worth adding beyond it.

There is no CMake project. Builds use the VS Code task (`clang++ -std=c++17`). `main.cpp` is empty. Tests live under `tests/`.

---

## What exists

### Tensor

`Tensor` stores `float` data in a contiguous `std::vector`, with shape, strides, element access, reshape, numel, and printing.

`TensorUtils` adds `zeros`, `ones`, `random_uniform`, mean, variance, min, max, flatten-shape, and shape assertions.

There are no elementwise ops, matrix multiply, broadcasting, views, or gradients. A new `Tensor` is zero-filled. Layers that own weights do not run an initialization scheme.

### Layers

Every layer implements `type()`, `name()`, `output_shape()`, `parameter_count()`, and `forward()`. Nothing implements `backward()`.

| Layer | Shape and parameter count | Forward |
| --- | --- | --- |
| ReLU, LeakyReLU, Sigmoid, Tanh, GELU | yes | yes |
| Softmax | yes (1D only) | yes |
| Flatten | yes | yes |
| Linear | yes, owns weights and bias | yes |
| Conv2D | yes, stores kernel, stride, padding | throws |
| MaxPool2D | yes | throws |
| BatchNorm1D, BatchNorm2D | yes, shape checks only | throws |
| Dropout | yes | throws |

`Linear` is a naive nested loop: `y = Wx + b` on a 1D vector. Conv2D does not store a weight tensor yet; it only stores geometry.

### Model

`sequential` can add layers, print names, and run `forward()` by feeding each layer's output into the next. It cannot remove a layer, collect parameters, save, or train.

### Design-time analysis

These run from layer metadata. They do not need a working forward pass.

- **Shape inference.** Walks the stack and records each output shape.
- **Architecture validator.** Rejects broken topologies: conv after linear, linear before flatten, channel mismatches, bad dimensions, flatten after softmax.
- **Parameter counter** and **model summary.** Per-layer shapes and parameter totals.
- **Layer factory.** Builds layers from strings such as `relu`, `linear 128`, `conv 3 16 3`, `pool 2`. Tracks the current shape so later layers can infer input size.
- **Architecture analyzer.** Scores a model, then prints suggestions, warnings, and parameter bottlenecks (for example, missing batch norm or dropout, sigmoid or tanh in hidden layers, a linear layer that dominates the parameter count).
- **FLOPs analyzer.** Partial. Linear FLOPs are counted. Conv2D uses `2 * K * K * H_out * W_out` and omits input and output channels. Softmax and MaxPool2D are still TODOs. `FLOPs::total_FLOPs` is declared and not defined. Batch-norm counts features, not spatial size.

### Tests

`tests/` covers the tensor, tensor utils, activations, linear, sequential, conv and pool shape logic, shape inference, parameter counts, summaries, the layer factory, and the architecture analyzer. Several of those tests check metadata, not numeric forward results, because conv, pool, batch norm, and dropout still throw.

---

## What is planned

The first milestone is a design studio: build a CNN or MLP, see shapes, catch invalid graphs, and read parameters, FLOPs, memory, and receptive field, with no training. The second milestone is a network that actually trains.

### Finish the design studio

1. **FLOPs analyzer.** Count conv as `2 * C_in * C_out * K * K * H_out * W_out`, plus pool, softmax, batch norm over the real tensor size, and the other activations. Implement `total_FLOPs` and return a per-layer breakdown instead of file-scope counters.
2. **Memory analyzer.** Parameter bytes, activation bytes per layer for a given batch size, and peak activation memory.
3. **Receptive-field analyzer.** Jump and receptive-field size after each conv and pool.
4. **Interactive builder.** A CLI in `main.cpp`: add and remove layers, print the architecture, run the analyzers, export the stack. `LayerFactory` is already the parser for this.

### Make forward real

Implement `forward()` for Conv2D, MaxPool2D, BatchNorm1D, BatchNorm2D, and Dropout. Conv should own a weight tensor and a bias. Batch norm needs scale, shift, and running mean and variance. Dropout needs a train versus eval flag, which `sequential` does not have yet.

Give every parameter tensor an initializer: zeros, uniform, normal, Xavier, He.

### Data, loss, and a training loop

The original roadmap sequences this as:

1. **Datasets.** A `Dataset` interface, then CIFAR-10, MNIST, Fashion-MNIST, and a folder or array loader. Batching belongs with the dataset, not inside `sequential`.
2. **Losses.** Cross-entropy first, then MSE, L1, and Huber.
3. **Backward.** A `backward()` on `Layer`, implemented for Linear, the activations, Conv2D, MaxPool2D, and both batch norms. See the autograd section below before committing to hand-written gradients only.
4. **Optimizers.** SGD, momentum, Adam. Each optimizer needs a parameter list, which `sequential` should expose.
5. **Trainer.** Epochs, batches, loss, accuracy, checkpoints, and an eval mode that turns dropout off and freezes batch-norm statistics.

### Performance, after the numerics are correct

OpenMP over independent output positions, SIMD on the inner loops, a memory pool so forward and backward stop allocating every call, cache-friendly conv (im2col plus a GEMM), then a CUDA backend, quantization, and mixed precision.

### Already named as later features

Residual blocks, weight serialization, and ONNX export. Residuals need a graph container; `sequential` cannot express a skip connection.

---

## What can be added

These are not in `PLAN.txt`. They are the pieces that turn the library from a layer list into a framework.

### Autograd engine

Stage 18 in `PLAN.txt` adds a hand-written `backward()` on each layer. That works for a fixed layer set. It does not scale: every new layer, loss, or fused op needs its own gradient, and a mistake is silent.

A reverse-mode autograd engine records primitive tensor ops on a tape and fills `.grad` on the way back. Layers become small graphs of those ops. A new activation is a few lines and gets a correct gradient for free.

Practical shape for this repo:

- `Tensor` gains `grad`, `requires_grad`, and a node id on the tape.
- The tape stores op type, parent ids, and the saved tensors a backward needs.
- First ops: add, multiply, matmul, relu, sum, log, exp, and a reduction for mean.
- `Linear`, the elementwise activations, and MSE or log-softmax cross-entropy are written as those ops.
- Conv2D, MaxPool2D, and BatchNorm stay fused ops with hand-written backward until im2col makes conv a matmul on the tape.
- `loss.backward()` walks the tape. `optimizer.step()` reads each parameter's `.grad`.
- A finite-difference gradient check in tests guards the fused ops.

Hand-written `Layer::backward()` can still exist as a thin wrapper that runs the tape, so the trainer does not care which style a layer uses.

Do this before writing twelve separate backward functions. The tensor op set is the dependency.

### Tensor algebra the engine needs

Broadcasting, matmul, batched matmul, transpose that can be a view, reductions, and im2col. Without these, both autograd and Conv2D forward stay special cases.

### Parameters and initialization

A `parameter()` list on `Layer` and `sequential`: every weight and bias, marked `requires_grad`. Initializers live next to that, so Linear and Conv2D stop starting at zero (zero weights make every ReLU unit identical).

### A module graph

Keep `sequential` for straight stacks. Add a small `Module` base and an `Add` or residual block so a skip is `y = F(x) + x`. Shape inference and the validator then walk a graph, not only a vector. This is the prerequisite for the residual blocks already listed in stage 22.

### More layers, once ops exist

Worth adding in roughly this order:

- AvgPool2D, global average pool, Conv1D
- LayerNorm (cheaper to autograd than BatchNorm)
- Embedding
- Concat and Add
- ConvTranspose2D
- Multi-head attention, only after batched matmul and softmax over an axis exist

### Data pipeline

`Dataset`, `DataLoader` (shuffle, batch, drop-last), a tensor collate, and light augmentation (flip, crop, normalize). MNIST is the right first file format; CIFAR-10 matches the conv stack.

### Training extras

Learning-rate schedules (step, cosine), weight decay, gradient clipping, early stopping, and a metric object for loss and accuracy. Checkpointing should save parameter tensors plus optimizer moments, not only the layer name list.

### Numerics and tests

`double` tensors, or a scalar type template, so gradient checks are stable. Compare Linear, ReLU, and a tiny conv against finite differences. A single overfitting test (one batch, loss goes to zero) is the signal that backward, loss, and the optimizer agree.

### Tooling

- A `CMakeLists.txt` with one library target and one test target. The current compile line globs `src/**/*.cpp` into every test.
- Export the architecture as JSON from the CLI (layer type, hyperparameters, inferred shapes, params, FLOPs).
- Later: a binary checkpoint format, then ONNX export of the sequential subset.

### Execution backends

Keep the naive loops as the reference. Add an OpenMP or SIMD backend behind the same op interface, and only then a CUDA backend. One op interface means autograd backward code does not fork per device.

---

## Suggested order

1. Tensor ops (add, mul, matmul, reductions) and initializers.
2. Autograd tape, with Linear and ReLU as the first clients, plus a gradient check.
3. Finish Conv2D, MaxPool2D, BatchNorm, and Dropout forward. Conv and pool backward can be fused ops.
4. Cross-entropy, SGD, and a trainer that overfits one batch.
5. MNIST loader, accuracy, checkpoints.
6. Close the design-studio gaps: FLOPs, memory, receptive field, CLI.
7. Adam, schedules, DataLoader, CMake.
8. Performance passes and a residual block, after the reference results are trustworthy.
