import torch
import torch.nn as nn

from data import X,y
from model import NonLinearClassifier

model = NonLinearClassifier()

X_batch = X[:32]
y_batch = y[:32]

loss_fn = nn.CrossEntropyLoss()

#Define SGD Optimizer 

optimizer = torch.optim.SGD(
    model.parameters(),
    lr = 0.1
)

#choose parameter to inspect

parameter = model.output_layer.bias

print("\n Parameter before update")
print(parameter.detach().clone())

#Forward propagation

logits = model(X_batch)

#calculate loss

loss = loss_fn(logits, y_batch)

print('\n Loss:')
print(loss.item())
loss.backward()
# calculate gradient
print('\nGradient')
print(parameter.grad)

# save parameter AND gradient before updating
parameter_before = parameter.detach().clone()
gradient_before = parameter.grad.detach().clone()

# update parameter
optimizer.step()

print('\n Parameter after update')
print(parameter.detach())

#Manually calculated the expected update
expected_parameter = parameter_before - 0.1 * gradient_before

print("\nExpected parameter according to SGD formula:")
print(expected_parameter)