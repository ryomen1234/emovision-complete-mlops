import torch
import torch.nn as nn 


class Model_0(nn.Module):

    def __init__(self, output: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, padding=1, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),

            nn.Conv2d(in_channels=16, out_channels=32, stride=1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*24*24, 64),
            nn.ReLU(),
            nn.Linear(64, output)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)