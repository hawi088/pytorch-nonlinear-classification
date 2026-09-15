import torch
import torch.nn as nn


from data import X, y
from model import NonLinearClassifier

model = NonLinearClassifier()
X_batch = X[:32]
y_batch = y[:32]
logits = model(X_batch)
logits.retain_grad()
print()
print()
print(logits)
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, y_batch)
print("Loss:", loss.item())

loss.backward()
print("\nGradient of loss with respect to logits:")
print(logits.grad)

print("\nOutput layer bias gradient:")
print(model.output_layer.bias.grad)
print("\nGradients AFTER backward:")

for name, parameter in model.named_parameters():
    print(name)
    print(parameter.grad)