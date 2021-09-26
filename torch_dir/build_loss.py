import torch

# from utils import label_to_id
import torch.nn as nn


def build_loss_func(config: dict, weight: list = None):
    criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss(weight=weight)
    # criterion = nn.CrossEntropyLoss(reduction="mean")
    # criterion = Ce_Bce_Loss(config.pairs, weight)

    return criterion


class Ce_Bce_Loss(nn.Module):
    def __init__(self, pairs: list, weight: list):
        super().__init__()
        self._pairs = pairs
        self._ce = nn.CrossEntropyLoss(weight=weight)
        self._bce_loss = nn.BCELoss()
        self._softmax = nn.Softmax(dim=1)

    def forward(self, outputs, labels):
        total_loss = self._ce(outputs, labels)
        outputs = self._softmax(outputs)
        self._bce_weight = 0.1

        for pair in self._pairs:
            label1, label2 = [torch.tensor(label_to_id(i)) for i in pair]
            index = labels == label1
            wrong_out = outputs[index, label2]
            if len(wrong_out) > 0:
                t = wrong_out.shape
                zeros_label = torch.zeros(len(wrong_out)).float()
                loss = self._bce_loss(wrong_out, zeros_label)
                total_loss += loss
            label1, label2 = label2, label1
            index = labels == label1
            wrong_out = outputs[index, label2]
            if len(wrong_out) > 0:
                t = wrong_out.shape
                zeros_label = torch.zeros(len(wrong_out)).float()
                loss = self._bce_loss(wrong_out, zeros_label)
                loss *= self._bce_weight
                total_loss += loss
        return total_loss
