#Core neural network architecture
import torch.nn as nn
class NonLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2,16)
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Linear(16,16)
        self.activation2 = nn.ReLU()
        self.output_layer = nn.Linear(16,2)
    def forward(self,x):
        x = self.layer1(x)
        x = self.activation1(x)

        x = self.layer2(x)
        x = self.activation2(x)
        x = self.output_layer(x)
        return x