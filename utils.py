import torch
import numpy as np


def IoU(y_true, y_pred, thr=0.5):
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred < 0.5] = 0

    I = (y_pred * y_true).sum()
    U = np.count_nonzero((y_pred + y_true))
    return I / U


class DiceCoeff(torch.nn.Module):
    def __init__(self):
        super(DiceCoeff, self).__init__()

    def forward(self, input, target):
        eps = 0.0001
        inter = torch.dot(input, target)
        union = torch.sum(input) + torch.sum(target) + eps

        t = 1 - (2 * inter.float() + eps) / union.float()
        return t


def dict_collate(batch):
    ret = {}
    elem = batch[0]

    # label
    label = []
    for i in range(len(batch)):
        label.append(batch[i]["label"])
    label = torch.cat(label, dim=0)
    ret["label"] = label

    for k in elem.keys():
        if k == "label":
            continue
        cur = []
        for i in range(len(batch)):
            cur.append(batch[i][k].unsqueeze(0))
        cur = torch.cat(cur, dim=0)
        ret[k] = cur
    return ret
