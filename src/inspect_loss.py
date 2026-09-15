import torch
import torch.nn.functional as F

logits = torch.tensor([2.3,-0.7])

probabilities = F.softmax(logits, dim=0)

print("Logits:")
print(logits)

print("\nProbabilities:")
print(probabilities)

print("\nSum:")
print(probabilities.sum())

print("\nPredicted class:")
print(torch.argmax(logits))