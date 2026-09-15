import torch
import torch.nn as nn

from data import X,y
from model import NonLinearClassifier

model = NonLinearClassifier()


#loss and optimizer 
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.01,
    weight_decay=0.001
)

#train

epochs = 100
for epoch in range (epochs):
    logits = model(X)
    loss=loss_fn(logits,y)
    optimizer.zero_grad
    loss.backward()
    optimizer.step()
    predictions = torch.argmax(logits, dim=1)

    accuracy = (predictions == y).float().mean() * 100

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1:3d} | "
            f"Loss: {loss.item():.4f} | "
            f"Accuracy: {accuracy.item():.2f}%"
        )
