import torch
import torch.nn as nn
import timm

def build_model(name: str, num_classes: int, pretrained: bool=True):
    model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    return model

class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    def forward(self, pred, target):
        n = pred.size(1)
        log_probs = pred.log_softmax(dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))
