import torch
import torch.nn as nn

from data import X, y
from model import NonLinearClassifier
from utils import create_initial_state


# -------------------------
# Reproducible initialization
# -------------------------

initial_state = create_initial_state()

model = NonLinearClassifier()
model.load_state_dict(initial_state)


# -------------------------
# Loss + Optimizer
# -------------------------

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001
)


# -------------------------
# Training
# -------------------------

epochs = 100

for epoch in range(epochs):

    # Forward
    logits = model(X)

    # Loss
    loss = loss_fn(logits, y)

    # Backward + update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Accuracy
    predictions = torch.argmax(logits, dim=1)
    accuracy = (predictions == y).float().mean() * 100

    # Report
    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1:3d} | "
            f"Loss: {loss.item():.4f} | "
            f"Accuracy: {accuracy.item():.2f}%"
        )