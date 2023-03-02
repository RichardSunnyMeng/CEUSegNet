import torch
import random
from dataset import CEUSdataset
from unet.unet_model import CEUSegNet
from torch.utils.data import DataLoader
import numpy as np

from config import *
from utils import DiceCoeff, IoU, dict_collate

class train_helper():
    def __init__(self, model=CEUSegNet):
        self.set_seed(2020)
        self.model = model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[40, 70], gamma=0.1)

        dataset_train = CEUSdataset(is_train=True, transform=None, datadir=r'..\data_CLMN')
        self.dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True,
                                      drop_last=False, collate_fn=dict_collate)

        dataset_test = CEUSdataset(is_train=False, transform=None, datadir=r'..\data_CLMN')
        self.dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, pin_memory=True,
                                          drop_last=False, collate_fn=dict_collate)

        self.loss = DiceCoeff()

        self.to_cuda()

        self.best_metric = None

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = False

    def to_cuda(self):
        self.model.to("cuda")
        self.model = torch.nn.DataParallel(self.model, device_ids=[0])

    def to_cpu(self):
        self.model = self.model.to("cpu")

    def get_device(self):
        if isinstance(self.model, torch.nn.DataParallel):
            module = self.model.module
        else:
            module = self.model
        for _, paras in module.named_parameters():
            return paras.device

    def train(self, epoch):
        self.model.train()
        for x in self.dataloader_train:
            x = CEUSdataset.toCuda(self.get_device(), x)
            y = self.model(x)
            l = self.loss(x["label"].reshape(-1), y["logit"].reshape(-1))
            l.backward()
            self.optimizer.step()
        self.scheduler.step()

    def test(self, epoch):
        self.model.eval()
        out = []
        gt = []
        l = 0
        for x in self.dataloader_train:
            x = CEUSdataset.toCuda(self.get_device(), x)
            y = self.model(x)
            l += self.loss(x['label'].reshape(-1), y["logit"].reshape(-1)).detach().cpu().numpy()
            out.extend(y['logit'].reshape(-1).detach().cpu().numpy())
            gt.extend(x['label'].reshape(-1).detach().cpu().numpy())
        self.print_info(out, gt, l, epoch)

        out = []
        gt = []
        l = 0
        for x in self.dataloader_test:
            x = CEUSdataset.toCuda(self.get_device(), x)
            y = self.model(x)
            l += self.loss(x['label'].reshape(-1), y["logit"].reshape(-1)).detach().cpu().numpy()
            out.extend(y['logit'].reshape(-1).detach().cpu().numpy())
            gt.extend(x['label'].reshape(-1).detach().cpu().numpy())
        self.print_info(out, gt, l, epoch)
        print("*******************************")

    def print_info(self, out, gt, loss, epoch, is_test=False, path="./"):
        out, gt = np.array(out), np.array(gt)
        iou = IoU(gt, out)
        print("epoch: {}, loss: {:.2f}, iou: {:.2f}".format(epoch, loss, iou))
        if is_test:
            if self.best_metric is None:
                self.best_metric = (epoch, iou)
            else:
                if iou > self.best_metric[1]:
                    self.best_metric = (epoch, iou)
                    self.save_model(iou, path=path)
                print("best model: epoch {}, auc {}".format(self.best_metric[0], self.best_metric[1]))

    def save_model(self, auc, path="./best_models"):
        if hasattr(self.model, "module"):
            torch.save(self.model.module.state_dict(), path + r"/best-{}.pth".format(auc))
        else:
            torch.save(self.model.state_dict(), path + r"/best-{}.pth".format(auc))

    def load_model(self, path):
        state = torch.load(path)
        if "cuda" in str(self.get_device()):
            self.model.module.load_state_dict(state)
        else:
            self.model.load_state_dict(state)


if __name__ == "__main__":
    helper = train_helper()
    for epoch in range(epoches):
        helper.train(epoch)
        helper.test(epoch)


