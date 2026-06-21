import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self, model, feat_dim=2048):
        super(ResNet, self).__init__()
        self.resnet = model
        self.resnet.fc = nn.Identity()
        self.resnet.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.inv_head = nn.Sequential(
            nn.Linear(feat_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256, bias=False)
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.inv_head(x)

        return x