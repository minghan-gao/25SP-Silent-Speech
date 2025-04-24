import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNWordClassifier(nn.Module):
    def __init__(self, seq_len: int, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.pool  = nn.MaxPool1d(2)    # halves length
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(2)

        flat_size = 32 * (seq_len // 4)  # two pools of 2
        self.fc1 = nn.Linear(flat_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (B,1,seq_len)
        x = self.pool (F.relu(self.conv1(x)))   # -> (B,16,seq_len/2)
        x = self.pool2(F.relu(self.conv2(x)))   # -> (B,32,seq_len/4)
        x = x.flatten(1)                        # -> (B, 32*(seq_len/4))
        x = F.relu(self.fc1(x))
        return self.fc2(x)
