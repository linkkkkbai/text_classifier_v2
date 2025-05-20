import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.ops as ops
import timm

class AttributeHead(nn.Module):
    def __init__(self, in_features, num_classes):
        """
        Initializes the AttributeHead module with two fully connected layers.
        
        Args:
            in_features (int): Number of input features.
            num_classes (int): Number of output classes.
        """
        super(AttributeHead, self).__init__()
        self.fc1 = nn.Linear(in_features, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ResNetCropMultiAttr(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        resnet = torchvision.models.resnet50(pretrained=cfg.model.pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # 去掉fc
        self.feature_dim = resnet.fc.in_features  # 2048 for resnet50

        self.type_head = nn.Linear(self.feature_dim, len(cfg.attributes.type.classes))
        self.font_head = nn.Linear(self.feature_dim, len(cfg.attributes.font.classes))
        self.italic_head = nn.Linear(self.feature_dim, len(cfg.attributes.italic.classes))

    def forward(self, x):
        feats = self.backbone(x)  # [N, 2048, 1, 1]
        feats = feats.flatten(1)
        return {
            'type': self.type_head(feats),
            'font': self.font_head(feats),
            'italic': self.italic_head(feats)
        }