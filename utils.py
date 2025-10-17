import bm3d
import ipdb
import yaml
import argparse
import openpyxl
import torch
import torchvision.transforms.functional as F
import numpy as np
import tifffile as tiff
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def standardized_visualization(x):
    assert x.dtype == np.uint16
    vrange = x.max() - x.min()

    if x.ndim == 3:
        y = np.zeros_like(x, dtype=np.float16)
        for i in range(x.shape[0]):
            y[i, ...] = (x[i, ...] - x[i, ...].min()) / (x[i, ...].max() - x[i, ...].min() + 1e-6) * vrange
    elif x.ndim == 2:
        y = x - x.min()
    else:
        raise ValueError("Unsupport input shape.")
    
    return np.clip(y, 0, vrange).astype(np.uint16)


def simpleNormalize(x, mode="hard"):
    if mode == "hard":
        p1, p2 = x.min(), x.max()
    elif mode == "soft":
        tmp = x.astype(np.float32).flatten()
        tmp.sort()
        p1 = tmp[int(len(tmp) * 0.001)]
        p2 = tmp[int(len(tmp) * 0.999)]

    x = (x - p1) / (p2 - p1 + 1e-8) 
    return x


def consistent_translate(x, y):
    origin_shape = x.shape

    x = x.astype(np.float32).flatten()
    y = y.astype(np.float32).flatten()

    X = np.vstack([x, np.ones(len(x))]).T
    a, b = np.linalg.lstsq(X, y, rcond=None)[0]

    x = (a * x + b).reshape(origin_shape)
    y = y.reshape(origin_shape)
    return x, y, (a, b)


def split_image(x):
    b, _, h, w = x.shape
    y1 = torch.FloatTensor(b, 1, h//2, w//2).to(x.device)
    y2 = torch.FloatTensor(b, 1, h//2, w//2).to(x.device)

    for ind in range(b):
        i = np.random.randint(2)
        j = np.random.randint(2)
        y1[ind, 0, :, :] = x[ind, 0, i::2, j::2]
        y2[ind, 0, :, :] = x[ind, 0, 1-i::2, 1-j::2]
    
    return y1, y2


def bm3d_denoise(x, sigma, p1=None, p2=None):
    dtype = x.dtype
    device = x.device

    if p1 is not None and p2 is not None:
        x = (x + 1) / 2 * (p2 - p1) + p1
    
    x = x.cpu().numpy().astype(np.uint16)
    y = np.zeros_like(x)
    for i in range(x.shape[0]):
        y[i, 0, :, :] = bm3d.bm3d(x[i, 0, :, :], sigma)

    if p1 is not None and p2 is not None:
        y = (y - p1) / (p2 - p1) * 2 - 1
    
    return torch.from_numpy(y).to(dtype).to(device)


def mse_with_gaussian_blur(image, target, sigma1=1.0, sigma2=2.0):
    k_size1 = int(2 * 4 * sigma1 + 1)
    k_size2 = int(2 * 4 * sigma2 + 1)

    image_blured = F.gaussian_blur(image, kernel_size=k_size1, sigma=sigma1).cpu().numpy()
    target_blured = F.gaussian_blur(target, kernel_size=k_size2, sigma=sigma2).cpu().numpy()
    image_aligned, target_aligned, _ = consistent_translate(image_blured, target_blured)

    return np.mean((image_aligned - target_aligned) ** 2)


def save_args_to_yaml(args, path):
    # 将Namespace转为dict并保存
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(vars(args), f, allow_unicode=True)


def load_args_from_yaml(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    parser = argparse.ArgumentParser()
    # 用yaml中的参数覆盖默认参数
    for k, v in data.items():
        if hasattr(parser.parse_args([]), k):
            parser.set_defaults(**{k: v})
    return parser.parse_args([])


class QueueList():
    def __init__(self, max_size=10):
        self.max_size = max_size
        self.list = []

    def add(self, x):
        if len(self.list) == self.max_size:
            self.list.pop(0)
        self.list.append(x)

    def remove(self, i):
        self.list.pop(i)


class XlsBook():
    def __init__(self, labels, sheet_name='log'):
        self.labels = labels
        self.book = openpyxl.Workbook()
        self.sheet = self.book.create_sheet(sheet_name, 0)
        self.sheet.append(labels)

    def write(self, values):
        if len(values) != len(self.labels):
            raise ValueError('Inputs of logger does not match the length of the labels.')
        self.sheet.append(values)

    def save(self, save_path):
        self.book.save(save_path)


if __name__ == '__main__':
    # stack = torch.tensor([[[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]],
    #                       [[-1,-2,-3,-4],[-5,-6,-7,-8],[-9,-10,-11,-12],[-13,-14,-15,-16]],
    #                       [[101,102,103,104],[105,106,107,108],[109,110,111,112],[113,114,115,116]]]])
    # print(stack.shape)
    # print(slice_fusion(stack))
    # print(slice_fusion(stack).shape)

    # stack = tiff.imread('data/ConvA/raw/conva.tif')[None, ...]
    # stack = torch.from_numpy(stack.astype(np.float32))
    # substack = stack[:, :2, ...]
    # fusedsub = slice_fusion(substack).numpy()[0].astype(np.uint16)
    # outp = np.concatenate((fusedsub, substack.numpy()[0].astype(np.uint16), fusedsub, fusedsub), axis=0)
    # tiff.imwrite('tmp.tif', outp)

    outp = tiff.imread('results/conva_base_09/epoch_200/001.tif')
    # denoised = outp[1]
    # # denoised = tiff.imread('data/ConvA/raw/conva.tif')[0]
    # gt = tiff.imread('data/ConvA/gt_single.tif')
    # score_ssim, score_psnr = compare(denoised, gt)
    # print(score_ssim, score_psnr)