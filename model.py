import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
            Flatten(),
            # nn.Linear(4608, 120),
            nn.Linear(1152, 120),
            nn.ReLU(True),
            nn.Linear(120, 84),
            nn.ReLU(True)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(84, 10),
        )
    
    def forward(self, x):
        feature = self.feature(x)       # Sequential_1
        pred = self.classifier(feature)       # Dropout -> Dense -> Activation
        return pred, feature


