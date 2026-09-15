import torch
from data import X, y
from model import NonLinearClassifier

model = NonLinearClassifier()
X_batch = X[:4]
y_batch = y[:4]

x,x1,a1,x2,a2,logits = model(X_batch)

print("Input:", x.shape)
print("After Linear 1:", x1.shape)
print("After ReLU 1:", a1.shape)
print("After Linear 2:", x2.shape)
print("After ReLU 2:", a2.shape)
print("Logits:", logits.shape)
print()
print()
print()
print("\nInput:")
print(x)

print("\nAfter Linear 1:")
print(x1)

print("\nAfter ReLU 1:")
print(a1)

print("\nAfter Linear 2:")
print(x2)

print("\nAfter ReLU 2:")
print(a2)

print("\nLogits:")
print(logits)