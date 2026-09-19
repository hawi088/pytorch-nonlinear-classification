import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from data import X,y
from utils import set_seed
#Reproducablity
set_seed(42)

#Train

indicies = torch.randperm(len(X)) #Randomization for training and test data

train_indicies = indicies[:800]
val_indicies = indicies[800:]
X_train = X[train_indicies]
y_train = y[train_indicies]

X_val = X[val_indicies]
y_val = y[val_indicies]
#Dataset
train_dataset = TensorDataset(X_train,y_train) #bundling
val_dataset = TensorDataset(X_val, y_val)

#DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True #order in which training data is presented
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))

X_batch, y_batch = next(iter(train_loader))

print("\nTraining batch:")
print("X batch shape:", X_batch.shape)
print("y batch shape:", y_batch.shape)

X_val_batch, y_val_batch = next(iter(val_loader))

print("\nValidation batch:")
print("X batch shape:", X_val_batch.shape)
print("y batch shape:", y_val_batch.shape)