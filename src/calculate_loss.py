import torch
import torch.nn as nn
from data import X, y
from model import NonLinearClassifier

model = NonLinearClassifier()

X_batch = X[:32]
y_batch = y[:32]

logits = model(X_batch)

loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, y_batch)
print(loss)
print("Logits shape:", logits.shape)
print("Targets shape:", y_batch.shape)
print("Loss:", loss.item())
