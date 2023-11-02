import random
import kornia
import torch
from torch.nn.functional import mse_loss
from module.gpu import *

# Numpy image -> batch tensor
def tensor(img, cpu=False):
    if len(img.shape) == 4:
        if cpu:
            return torch.FloatTensor(img.transpose(0, 3, 1, 2))
        else:
            return torch.FloatTensor(img.transpose(0, 3, 1, 2)).to(device)
    else:
        if cpu:
            return torch.FloatTensor(img.transpose(2, 0, 1)).unsqueeze(0)
        else:
            return torch.FloatTensor(img.transpose(2, 0, 1)).unsqueeze(0).to(device)

def mse(A, B, use_tensor=False):
    if use_tensor:
        # mse = mse_loss(A, B, reduction='none')
        mse = ((A - B)**2).mean(-1).mean(-1).mean(-1)
    else:
        mse = ((A - B)**2).mean()
    return mse

def sliding_window(image, step_size, window_size, no_dup=False):
    hyperbreak = False
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            if y + window_size[0] < image.shape[0]:
                if x + window_size[1] < image.shape[1]:
                    yield x, y, image[y:y + window_size[0], x:x + window_size[1]]
                else:
                    if no_dup:
                        break
                    yield x, y, image[y:y + window_size[0], image.shape[1] - window_size[1]:]
                    break
            else:
                if x + window_size[1] < image.shape[1]:
                    yield x, y, image[image.shape[0] - window_size[0]:, x:x + window_size[1]]
                else:
                    if no_dup:
                        break
                    yield x, y, image[image.shape[0] - window_size[0]:, image.shape[1] - window_size[1]:]
                    hyperbreak = True
                    break
        if hyperbreak:
            break

def num_sliding_windows(image, step_size, window_size):
    hyperbreak = False
    cnt = 0
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            if y + window_size[0] < image.shape[0]:
                if x + window_size[1] < image.shape[1]:
                    cnt += 1
                else:
                    cnt += 1
                    break
            else:
                if x + window_size[1] < image.shape[1]:
                    cnt += 1
                else:
                    cnt += 1
                    hyperbreak = True
                    break
        if hyperbreak:
            break
    return cnt

def shuffled_sliding_window(image, pair, step_size, window_size):
    x_list = [i for i in range(0, image.shape[1] - window_size[1], step_size)]
    y_list = [i for i in range(0, image.shape[0] - window_size[0], step_size)]
    if image.shape[1] - window_size[1] not in x_list:
        x_list.append(image.shape[1] - window_size[1])
    if image.shape[0] - window_size[0] not in y_list:
        y_list.append(image.shape[0] - window_size[0])
    
    xy_list = []
    i = 0
    for y in y_list:
        for x in x_list:
            xy_list.append((i, x, y))
            i += 1

    np.random.shuffle(xy_list)

    for i, x, y in xy_list:
        if y + window_size[0] < image.shape[0]:
            if x + window_size[1] < image.shape[1]:
                yield i, x, y, image[y:y + window_size[0], x:x + window_size[1]], pair[y:y + window_size[0], x:x + window_size[1]]
            else:
                yield i, x, y, image[y:y + window_size[0], image.shape[1] - window_size[1]:], pair[y:y + window_size[0], image.shape[1] - window_size[1]:]
        else:
            if x + window_size[1] < image.shape[1]:
                yield i, x, y, image[image.shape[0] - window_size[0]:, x:x + window_size[1]], pair[image.shape[0] - window_size[0]:, x:x + window_size[1]]
            else:
                yield i, x, y, image[image.shape[0] - window_size[0]:, image.shape[1] - window_size[1]:], pair[image.shape[0] - window_size[0]:, image.shape[1] - window_size[1]:]

def random_window(image, window_size, num_windows):
    for i in range(int(num_windows)):
        x = random.randint(0, image.shape[1]-window_size[1]-1)
        y = random.randint(0, image.shape[0]-window_size[0]-1)
        yield x, y, image[y:y + window_size[0], x:x + window_size[1]]
