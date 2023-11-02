import kornia
import numpy as np
import cv2
import math
from env.utils import sliding_window, mse

def calculate_complexity(img, use_tensor=True):
    if use_tensor:
        lab = kornia.color.rgb_to_lab(img)
        gxy = kornia.filters.sobel(lab)
        complexity = gxy.max(1)[0].mean(-1).mean(-1)
        return complexity.view(-1, 1)
    else:
        img = (img*255).astype(np.uint8)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        gx = cv2.Sobel(lab, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(lab, cv2.CV_32F, 0, 1)
        gmax = np.sqrt(gx**2 + gy**2).max(-1)
        complexity = gmax.mean()
        return complexity

def get_complexity_heatmap(img, step_size, window_size, norm=False, minmaxnorm=False, standard=False, use_tensor=False, blur=False):
    complexities = []
    ch = 0
    for x, y, part in sliding_window(img, step_size=step_size, window_size=window_size):
        if x == 0:
            ch += 1
        complexities.append(calculate_complexity(part, use_tensor=False))
    cw = int(len(complexities)/ch)
    if norm:
        img_complexity = calculate_complexity(img, use_tensor=False) + 1
        complexities /= np.array(img_complexity)
    if minmaxnorm:
        if complexity_heatmap.max() - complexity_heatmap.min() != 0:
            complexity_heatmap = (complexity_heatmap - complexity_heatmap.min()) / (complexity_heatmap.max() - complexity_heatmap.min())
        else:
            complexity_heatmap = np.zeros_like(complexity_heatmap)
    elif standard:
        mean = np.mean(complexities)
        min_comp = np.min(complexities)
        std = np.std(complexities)
        standard_complexities = []
        for comp in complexities:
            standard_comp = (comp - mean) / (std + 1e-12)
            standard_complexities.append(standard_comp)
        complexities = standard_complexities
    complexity_heatmap = np.reshape(complexities, (ch, cw))
    if blur:
        complexity_heatmap = cv2.GaussianBlur(complexity_heatmap, (int(window_size[0]/8)+1, int(window_size[0]/8)+1), 0)
    return complexity_heatmap

def get_comp2sim(C, G, complexity, use_tensor=False):
    sim = 1 - mse(C, G, use_tensor=use_tensor)
    # comp2sim = sim / (complexity + 1e-12)  # provent 0 division
    comp2sim = sim / ((math.tanh(complexity)*0.5)+0.5)
    return comp2sim


def calculate_ssim(a, b):
    ssim = kornia.metrics.ssim(a, b, window_size=11, max_val=1.0, eps=1e-12, padding='same')
    return ssim.mean(1).mean(1).mean(1)

def calculate_psnr(a, b):
    psnr = kornia.metrics.psnr(a, b, max_val=1)
    return psnr