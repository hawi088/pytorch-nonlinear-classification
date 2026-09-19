import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from data import X,y
from model import NonLinearClassifier
from utils import set_seed

#Reproducablity
set_seed(42)

#Train/validation split

indices = torch.randperm(len(X))

train_indices = indices[:800]
val_indices = indices[800:]

X_train = X[train_indices]
y_train = y[train_indices]

X_val = X[val_indices]
y_val= y[val_indices]

train_dataset = TensorDataset(X_train,y_train)
val_dataset = TensorDataset(X_val,y_val)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

model = NonLinearClassifier()
#loss and optimizer 
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

#Training Loop

epochs = 20

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    total_correct = 0
    for X_batch,y_batch in train_loader:
        #Forward
        logits = model(X_batch)
        #Loss
        loss = loss_fn(logits,y_batch)
        #backward + update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item() * X_batch.size(0)

        predictions = torch.argmax(logits, dim=1)
        total_correct += (predictions == y_batch).sum().item()


    # Epoch metrics
    train_loss = total_loss / len(train_dataset)
    train_accuracy = total_correct / len(train_dataset) * 100

    print(
        f"Epoch {epoch + 1:2d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2f}%"
    )