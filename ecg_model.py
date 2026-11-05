import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, k=7, p_drop=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, k, stride=stride, padding=k // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(cout)
        self.drop = nn.Dropout(p_drop)
        self.conv2 = nn.Conv1d(cout, cout, k, padding=k // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.Conv1d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm1d(cout))
        else:
            self.short = nn.Identity()

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.short(x))


class ResNet1D(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True), nn.MaxPool1d(2))
        self.blocks = nn.Sequential(
            ResBlock(32, 32), ResBlock(32, 64, stride=2), ResBlock(64, 64),
            ResBlock(64, 128, stride=2), ResBlock(128, 128), ResBlock(128, 128, stride=2))
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        h = self.blocks(self.stem(x))
        h = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)  # avg + max pooling over time
        return self.fc(self.drop(h))


CLASS_NAMES = ['Normal', 'AF', 'Other', 'Noisy']


def load_model(path='model_v4.pt'):
    model = ResNet1D()
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model