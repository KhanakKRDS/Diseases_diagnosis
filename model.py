import torch # PyTorch library (Handles tensors, GPU acceleration, math operations)
import torch.nn as nn # Neural network module (Defines layers, loss functions,nn.Linear, nn.Conv2d,nn.ReLU)
import torch.optim as optim # Optimization algorithms (SGD, Adam)
from torchvision import datasets, transforms # Datasets and image transformations
from torch.utils.data import DataLoader # Data loading and batching
import os # Operating system interfaces (File paths, directory management)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transforms.Resize((128, 128))
batch_size = 8

class DiseaseDiagnosisModel(nn.Module):
    def __init__(self, num_classes):
        super(DiseaseDiagnosisModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x