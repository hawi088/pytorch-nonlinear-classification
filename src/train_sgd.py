import torch
import torch.nn as nn

from data import X,y
from model import NonLinearClassifier

model = NonLinearClassifier()

#loss and optimizer

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr = 0.1
)

#training

epochs = 100
for epoch in range(epochs):
    #forward
    logits = model(X)
    #loss
    loss = loss_fn(logits,y)
    #clear old gradient
    optimizer.zero_grad()
    #backward
    loss.backward()
    #update the parameter
    optimizer.step()
    #Accuracy
    predictions = torch.argmax(logits,dim=1)
    accuracy = (predictions == y).float().mean() *100

    if (epoch+1)%10 == 0:
       print(
            f"Epoch {epoch + 1:3d} | "
            f"Loss: {loss.item():.4f} | "
            f"Accuracy: {accuracy.item():.2f}%"
)
    