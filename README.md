# Two Moons — Non-Linear Classification with PyTorch

A deeper, more realistic step up from XOR: 1000 points arranged as two interleaving crescents (`sklearn.datasets.make_moons`), classified with a 3-layer PyTorch network, and used as a testbed to compare optimizers (SGD, SGD+Momentum, Adam, AdamW) and to inspect exactly what's happening at each stage of forward pass, loss, and backpropagation.

Where the XOR project asked "can a hidden layer solve something a straight line can't," this project asks: "given that it can, how much does *how you train it* matter?"

## The Data

`make_moons(n_samples=1000, noise=0.15)` — two curved, interleaving clusters, 500 points each, with enough added noise that some points near the boundary genuinely overlap. Unlike XOR's 4 clean points, this can't be memorized — the model has to learn an actual curved decision region.

## The Model

```python
class NonLinearClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 16)
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Linear(16, 16)
        self.activation2 = nn.ReLU()
        self.output_layer = nn.Linear(16, 2)
```
2 inputs → 16 hidden units → 16 hidden units → 2 output classes. Deeper and wider than the XOR network, needed for a genuinely curved boundary instead of a simple corner.

## Files

| File | Description |
|---|---|
| `data.py` | Generates and visualizes the two-moons dataset |
| `model.py` | The `NonLinearClassifier` architecture |
| `utils.py` | `set_seed()` and `create_initial_state()` — for reproducible comparisons across optimizers |
| `train_sgd.py` | Trains with plain SGD |
| `train_sgd_momentum.py` | Trains with SGD + momentum (0.9) |
| `train_adam.py` | Trains with Adam |
| `train_adamw.py` | Trains with AdamW |
| `train_pipeline.py` | Demonstrates the train/val split + `DataLoader` batching setup |
| `train_loop.py` | Full training loop: mini-batches, train/val split, per-epoch validation metrics |
| `calculate_loss.py` | Minimal script showing a single forward pass + loss calculation |
| `inspect_forward.py` | Meant to print every intermediate tensor in the forward pass — **currently broken, see below** |
| `inspect_backward.py` | Prints every parameter's gradient after `loss.backward()` |
| `inspect_loss.py` | Standalone demo of how Softmax converts logits into probabilities |
| `inspect_optimizer.py` | Manually verifies PyTorch's SGD update against the SGD formula by hand |
| `compare_optimizers.py` | *(added)* Trains all 3 optimizers from identical initial weights, plots loss/accuracy side by side, and plots the final decision boundary |

## A Real Bug, Found While Reviewing

`inspect_forward.py` tries to unpack 6 return values from the model:
```python
x, x1, a1, x2, a2, logits = model(X_batch)
```
But `model.py`'s `forward()` only returns one value — the final `logits`:
```python
def forward(self, x):
    x = self.layer1(x)
    x = self.activation1(x)
    x = self.layer2(x)
    x = self.activation2(x)
    x = self.output_layer(x)
    return x
```
Running it throws `ValueError: not enough values to unpack (expected 6, got 4)`. The intent behind `inspect_forward.py` is clearly to show every intermediate step of the forward pass (input → after layer 1 → after ReLU → after layer 2 → after ReLU → logits) — but `forward()` needs to actually return all of those intermediate tensors as a tuple for that script to work:

```python
def forward(self, x):
    x1 = self.layer1(x)
    a1 = self.activation1(x1)
    x2 = self.layer2(a1)
    a2 = self.activation2(x2)
    logits = self.output_layer(a2)
    return x, x1, a1, x2, a2, logits
```
Note this would change the model's return signature everywhere it's used (`train_sgd.py`, `train_loop.py`, etc. all currently expect `logits = model(X)` directly) — so this is best kept as a separate debug-only version of `forward()`, or the calling scripts need `logits = model(X_batch)[-1]` instead.

## What Actually Works: The Optimizer Comparison

All three optimizers were trained from **identical initial weights** (via `create_initial_state()`), on the full dataset, for 100 epochs — so the only variable that changes between runs is the update rule itself.

| Optimizer | Final Loss | Final Accuracy |
|---|---|---|
| Plain SGD (lr=0.1) | 0.289 | 86.8% |
| SGD + Momentum (lr=0.1, momentum=0.9) | 0.070 | 98.2% |
| Adam (lr=0.01) | 0.052 | 98.3% |
| AdamW (lr=0.01) | 0.053 | 98.2% |

![Optimizer comparison](optimizer_comparison.png)

Momentum and Adam converge almost identically here and both comfortably beat plain SGD — plain SGD is still visibly climbing at epoch 100 rather than having converged. AdamW essentially matches Adam, which makes sense: AdamW only differs from Adam in how it applies weight decay, and no weight decay was configured here, so the two are mathematically doing the same thing in this setup.

## The Learned Decision Boundary

![Decision boundary](decision_boundary.png)

Worth noting: the boundary is made of straight-line segments, not smooth curves. That's not a training issue — it's a direct consequence of using ReLU activations. A network built entirely from ReLUs can only ever produce a **piecewise-linear** decision boundary, no matter how well-trained it is. What looks like a curve from a distance is actually many small linear segments stitched together — more hidden units means more segments, and therefore a boundary that can approximate smoother curves more closely.

## Proof: The Manual SGD Update Matches PyTorch's, Exactly

`inspect_optimizer.py` picks a single parameter, records it and its gradient before calling `optimizer.step()`, then compares PyTorch's actual updated value against the SGD formula computed by hand:

```
Parameter before update:  tensor([0.1900, 0.1392])
Gradient:                 tensor([ 0.1074, -0.1074])
Parameter after update:   tensor([0.1793, 0.1500])
Expected (manual formula): tensor([0.1793, 0.1500])
```

Exact match. `optimizer.step()` for plain SGD really is just `param -= lr * grad`, nothing hidden.

## The Full Training Pipeline (`train_loop.py`)

Unlike the other training scripts (which train on the full dataset at once), `train_loop.py` does it properly: an 800/200 train/val split, mini-batches of 32 via `DataLoader`, and per-epoch validation metrics — the version of this project closest to how real training actually works.

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 1 | 0.442 | 79.1% | 0.337 | 85.0% |
| 5 | 0.150 | 93.6% | 0.145 | 93.0% |
| 10 | 0.034 | 98.9% | 0.037 | 99.0% |
| 20 | 0.014 | 99.5% | 0.049 | 99.0% |

Worth noticing: training loss keeps dropping smoothly, but validation loss gets noisier and even ticks back up at times (epoch 11–12, 15, 19) while validation accuracy stays flat around 98–99%. That's an early, mild signature of the model starting to fit noise in the training batches rather than learning anything new — normal at this scale, and a natural segue into regularization if this network were pushed further.

## Requirements

```
torch
scikit-learn
matplotlib
numpy
```
```bash
pip install torch scikit-learn matplotlib numpy
```

## How to Run

```bash
python data.py                    # visualize the raw dataset
python train_sgd.py                # plain SGD baseline
python train_sgd_momentum.py       # + momentum
python train_adam.py               # Adam
python train_adamw.py              # AdamW
python train_loop.py               # full pipeline: mini-batches + train/val split
python inspect_optimizer.py        # verify SGD's update formula by hand
python compare_optimizers.py       # generates optimizer_comparison.png + decision_boundary.png
```

`inspect_forward.py` needs the `forward()` fix described above before it will run.

## What This Project Demonstrates

- That optimizer choice can matter as much as model architecture: identical weights, identical data, and the accuracy gap between plain SGD and Adam after 100 epochs is over 11 points
- Why ReLU networks always produce piecewise-linear decision boundaries, regardless of how well-trained they are
- That PyTorch's `optimizer.step()` is not a black box — for plain SGD, it's provably just `param -= lr * grad`, verified by hand
- The real difference between full-batch training (all 4 optimizer scripts) and proper mini-batch training with a validation split (`train_loop.py`) — and what an early overfitting signal actually looks like in the numbers
- That reproducible experiments require controlling initialization (`create_initial_state()`) — without it, "optimizer A beat optimizer B" could just mean "optimizer A got luckier weights"
