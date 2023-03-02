import torch
import torchvision
from torch.utils.data.dataset import Dataset
import os
import glob
import PIL.Image as Image
import cv2 as cv
import config
import numpy as np

class CEUSdataset(Dataset):
    def __init__(self, datadir=r'L:\mzl\CEUSSegNet\data_BC', is_train=True, transform=None):
        self.is_train = is_train
        self.transform = transform
        self.inner_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(config.size)
            ]
        )
        if "CLMN" in datadir:
            self.us_norm = torchvision.transforms.Normalize(mean=(0.21885207), std=(0.08665463))
            self.ceus_norm = torchvision.transforms.Normalize(mean=(0.1600279), std=(0.08724713))
        if "BC" in datadir:
            self.us_norm = torchvision.transforms.Normalize(mean=(0.25341047), std=(0.06942732))
            self.ceus_norm = torchvision.transforms.Normalize(mean=(0.12733308), std=(0.09379604))

        path = glob.glob(os.path.join(datadir, "*-*"))
        path.sort(key=lambda x: int(x.split(os.path.sep)[-1].split("-")[0]))

        if is_train:
            self.path = [path[i] for i in range(len(path)) if i % 3 != 0]
        else:
            self.path = [path[i] for i in range(len(path)) if i % 3 == 0]


    def __len__(self):
        return len(self.path)

    def __getitem__(self, item):
        path = self.path[item]

        us_path = os.path.join(path, 'US.jpg')
        ceus_path = os.path.join(path, "img.jpg")
        mask_path = os.path.join(path, "mask.jpg")

        us_img = cv.imread(us_path, cv.IMREAD_GRAYSCALE)
        us_img = Image.fromarray(us_img)
        us_img = self.inner_transform(us_img)

        ceus_img = cv.imread(ceus_path, cv.IMREAD_GRAYSCALE)
        ceus_img = Image.fromarray(ceus_img)
        ceus_img = self.inner_transform(ceus_img)

        mask_img = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        mask_img = Image.fromarray(mask_img)
        mask_img = self.inner_transform(mask_img)

        us_img = us_img.unsqueeze(0)
        ceus_img = ceus_img.unsqueeze(0)
        mask_img = mask_img.unsqueeze(0)

        us_img = self.us_norm(us_img)
        ceus_img = self.ceus_norm(ceus_img)

        imgs = torch.cat([us_img, ceus_img, mask_img], dim=0)

        if not self.transform is None:
            imgs = self.transform(imgs)

        return {"us": imgs[0], "ceus": imgs[1], "label": imgs[2]} # C * W * H (C = 1)

    @staticmethod
    def toCuda(device, ret):
        for k, v in ret.items():
            ret[k] = v.to(device)
        return ret


