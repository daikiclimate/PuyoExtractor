import torch
# from utils import label_to_id
import torch.nn as nn


def build_loss_func(config: dict):
    # criterion = nn.MSELoss()
    # criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    criterion = MseLoss()

    return criterion


class MseLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._mse = nn.MSELoss()
        self._mse = nn.L1Loss()

    def forward(self, anchor, outputs, labels):
        bs = outputs.shape[0]
        diff = torch.ones(bs).float().to(outputs.device).reshape(bs, 1)
        loss = self._mse(outputs - labels, diff)
        return loss
