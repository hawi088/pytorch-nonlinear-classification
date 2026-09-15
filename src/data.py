import torch
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
torch.manual_seed(42)
X,y = make_moons(
    n_samples = 1000,
    noise = 0.15,
    random_state = 42
)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

print("X shape:", X.shape)
print("y shape:", y.shape)
print()
print("X sample:", X[:5])
print("y sample:", y[:5])

print()
print("class 0:",(y == 0).sum().item())
print("class 1:",(y == 1).sum().item())

plt.figure(figsize=(8, 6))

plt.scatter(
    X[y == 0, 0],
    X[y == 0, 1],
    label="Class 0"
)

plt.scatter(
    X[y == 1, 0],
    X[y == 1, 1],
    label="Class 1"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Two-Moons Dataset")
plt.legend()

plt.show()