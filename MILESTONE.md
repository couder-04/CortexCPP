# CortexCPP milestones

Each milestone is one public update (a LinkedIn post plus the matching repo state). Post it only after every box in that milestone is done and the demo command works on a clean clone.

`PLAN.md` is the technical plan. This file is only the publish schedule.

The repo today can infer shapes, validate a stack, count parameters, and score an architecture. `main.cpp` is empty, the README is one line, and there is no build file outside VS Code. Empty tracked files (`Adam`, `SGD`, `Momentum`, `CrossEntropyLoss`, `CIFAR10Loader`, `BuilderCLI`, `MemoryAnalyzer`, `ReceptiveFieldAnalyzer`) and compiled test binaries are still in git. That is not a milestone. No post yet.

---

## Update 1 — Architecture engine

Post when a stranger can build a small conv net from the README and see the same summary you screenshot.

Demo: input `(1, 28, 28)` → Conv → ReLU → MaxPool → Flatten → Linear. The program prints the model summary (output shape and parameter count per layer, total parameters, valid-architecture line) and the architecture-analyzer score.

- [ ] `main.cpp` or `examples/` builds that network through `LayerFactory` and prints the summary and the analyzer score
- [ ] Shapes and parameter totals checked by hand against the screenshot
- [ ] Makefile or `CMakeLists.txt` builds the demo and the existing tests with one command
- [ ] README: what CortexCPP is, the build command, the demo command, and the pasted summary
- [ ] Tests for shape inference, parameter count, validation, and summary on that same network, and they pass
- [ ] Removed from git: compiled binaries, `.DS_Store`, and every zero-line placeholder (`Adam`, `SGD`, `Momentum`, `CrossEntropyLoss`, `CIFAR10Loader`, the CLI files, `MemoryAnalyzer`, `ReceptiveFieldAnalyzer`)

Say this: a C++ library that stacks conv and linear layers, infers every output shape, rejects an invalid topology, and counts parameters.

Leave for a later post: FLOPs, conv and pool forward, training, datasets, CUDA.

---

## Update 2 — A real forward pass

Post when the same small network runs on a tensor and prints numeric output, including conv and pool.

Demo: one input tensor through Conv → ReLU → MaxPool → Flatten → Linear, then print the output shape and a few output values.

- [ ] `forward()` implemented for Conv2D, MaxPool2D, BatchNorm1D, BatchNorm2D, and Dropout
- [ ] Conv2D owns a weight tensor and a bias; batch norm owns scale, shift, and running statistics
- [ ] Weights use an initializer (He or Xavier), so they are not left at zero
- [ ] `sequential` has a train versus eval switch; dropout is off in eval
- [ ] A test checks conv and pool output shape and a known tiny numeric case
- [ ] README demo runs this forward and shows the printed tensor

Say this: the architecture engine now executes the stack, including convolution and pooling.

Leave for a later post: FLOPs and memory tables, gradients, a training loop.

---

## Update 3 — Design studio

Post when the summary grows into a full static report and a person can build the net from a prompt.

Demo: the Update 2 network, plus a report with shapes, parameters, FLOPs, activation memory, and receptive field. A short CLI session adds a layer, removes a layer, and prints that report.

- [ ] FLOPs per layer are correct, including `2 * C_in * C_out * K * K * H_out * W_out` for conv, and real counts for pool, softmax, batch norm, and the other activations
- [ ] Memory analyzer: parameter bytes, activation bytes, and peak activation memory for a chosen batch size
- [ ] Receptive-field analyzer: jump and receptive-field size after each conv and pool
- [ ] CLI: add layer, remove layer, print architecture, run the report, export the stack (plain text or JSON)
- [ ] README shows the report and the CLI commands

Say this: CortexCPP can design a CNN, tell you what it costs, and export the architecture. Training is still ahead.

Leave for a later post: loss, backward, accuracy numbers.

---

## Update 4 — Autograd

Post when a loss on Linear and ReLU produces gradients that match a finite-difference check.

Demo: a two-layer MLP, a scalar loss, `backward()`, and a printed gradient check (analytic gradient next to a numeric gradient).

- [ ] Tensor ops used by the demo: add, multiply, matmul, relu, and a reduction
- [ ] Reverse-mode tape: `Tensor` carries `grad` and `requires_grad`; `backward()` fills parameter gradients
- [ ] Linear and the elementwise activations run as those ops
- [ ] A test compares their gradients to finite differences
- [ ] `sequential` exposes a parameter list the tape can see

Conv, pool, and batch norm may stay without gradients in this update. The post shows the MLP check only.

Say this: CortexCPP has a small autograd engine, and the first gradients are checked.

Leave for a later post: an optimizer stepping those gradients, a dataset, an accuracy number.

---

## Update 5 — It trains

Post when a tiny network overfits one batch and you can show the loss going down.

Demo: one fixed batch, SGD, cross-entropy or MSE, loss printed each step until it is near zero. Eval mode is on for the final forward.

- [ ] Cross-entropy (and MSE if the demo uses it)
- [ ] Conv2D, MaxPool2D, and batch norm have backward, either on the tape or as fused ops with a gradient check
- [ ] SGD updates the parameter list
- [ ] A training loop runs epochs over that one batch and records loss
- [ ] The overfitting run is a test or a documented command, and the loss curve is what you screenshot

Say this: the framework trains. A memorized batch is the proof, because that isolates backward, loss, and the optimizer.

Leave for a later post: a dataset accuracy, Adam, learning-rate schedules.

---

## Update 6 — A number you can quote

Post when a real dataset produces an accuracy you would say out loud.

Demo: MNIST (or CIFAR-10) through the trainer, with train loss and test accuracy. The README states the command, the epoch count, and the accuracy you got.

- [ ] Dataset loader and a `DataLoader` (shuffle, batch)
- [ ] Accuracy metric, eval mode, and a saved checkpoint (weights plus enough optimizer state to resume)
- [ ] Adam, plus either weight decay or a learning-rate schedule
- [ ] The quoted run is repeatable from the README

Say this: the accuracy, the dataset, and that the training loop is written in this library.

Leave for a later post: speedups, CUDA, ONNX, residual benchmarks. Quote a speed or an export only after that run exists.

---

## Later updates

Post one of these only when its demo is a number or a file someone else can open.

| Update | Post after | Demo |
| --- | --- | --- |
| Faster conv | OpenMP or SIMD, or im2col plus a matrix multiply, with the naive loop kept as the reference | Same net, two timings, same numeric result |
| Residual block | A module graph that can express `y = F(x) + x`, with shape inference and a forward | A small residual stack and its summary |
| Checkpoint reload | Load a saved run and match the previous eval accuracy | Before and after reload |
| CUDA | One op backend on GPU, same gradient check as CPU | One op's CPU and GPU outputs |
| ONNX | Export the sequential subset and open it in another runtime | The exported file and a matching output |

---

## Order

Update 1 is the first post. Updates 2 and 3 can swap if the forward pass lands before the design-studio report. Update 4 comes before Update 5. Update 6 comes after a trainer exists. The later table is unordered; ship whichever demo is finished.
