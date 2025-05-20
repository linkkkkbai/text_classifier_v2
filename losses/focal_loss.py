import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # 可以是 None 或 list/ndarray/torch.Tensor
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p = torch.exp(-ce_loss)
        loss = (1 - p) ** self.gamma * ce_loss

        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                alpha = torch.tensor(self.alpha, device=inputs.device)
            elif isinstance(self.alpha, torch.Tensor):
                alpha = self.alpha.to(inputs.device)
            else:
                alpha = torch.full((inputs.size(1),), self.alpha, device=inputs.device)
            at = alpha[targets]
            loss = at * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        


class MultiAttributeFocalLoss(nn.Module):
    def __init__(self, cfg):
        # super(MultiAttributeFocalLoss, self).__init__()
        # self.type_focal_loss = FocalLoss(
        #     gamma=cfg.training.focal_loss.gamma,
        #     alpha=cfg.attributes.type.weights,
        #     reduction='none'
        # )
        # self.font_focal_loss = FocalLoss(
        #     gamma=cfg.training.focal_loss.gamma,
        #     alpha=cfg.attributes.font.weights,
        #     reduction='none'
        # )
        # self.italic_focal_loss = FocalLoss(
        #     gamma=cfg.training.focal_loss.gamma,
        #     alpha=cfg.attributes.italic.weights,
        #     reduction='none'
        # )
        super().__init__()
        self.type_loss = nn.CrossEntropyLoss(reduction='none')
        self.font_loss = nn.CrossEntropyLoss(reduction='none')
        self.italic_loss = nn.CrossEntropyLoss(reduction='none')
    def forward(self, predictions, targets):

        type_loss = self.type_loss(predictions['type'], targets['type'])
        font_loss = self.font_loss(predictions['font'], targets['font']) 
        italic_loss = self.italic_loss(predictions['italic'], targets['italic'])

        type_loss = type_loss.mean()
        font_loss = font_loss.mean()
        italic_loss = italic_loss.mean() 

        # total_loss = type_loss + font_loss + italic_loss
        total_loss = type_loss
        losses = {
            'type': type_loss,
            'font': font_loss,
            'italic': italic_loss,
            'total': total_loss
        }
        return total_loss, losses
    



class MultiAttributeLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.type_loss = nn.CrossEntropyLoss(reduction='none')
        self.font_loss = nn.CrossEntropyLoss(reduction='none')
        self.italic_loss = nn.CrossEntropyLoss(reduction='none')
    def forward(self, predictions, targets):
        # targets: [B, 3] tensor
        if isinstance(targets, dict):
            type_targets = targets['type']
            font_targets = targets['font']
            italic_targets = targets['italic']
        else:
            # targets: [B, 3]
            type_targets = targets[:, 0]
            font_targets = targets[:, 1]
            italic_targets = targets[:, 2]

        type_loss = self.type_loss(predictions['type'], type_targets).mean()
        font_loss = self.font_loss(predictions['font'], font_targets).mean()
        italic_loss = self.italic_loss(predictions['italic'], italic_targets).mean()
        total_loss = type_loss + font_loss + italic_loss
        losses = {
            'type': type_loss,
            'font': font_loss,
            'italic': italic_loss,
            'total': total_loss
        }
        return total_loss, losses