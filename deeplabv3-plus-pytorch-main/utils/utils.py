import os
import random

import numpy as np
import torch
from PIL import Image


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def resize_image(image, size):
    iw, ih = image.size
    w, h = size

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    return new_image, nw, nh


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def preprocess_input(image):
    image /= 255.0
    return image


def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)


def download_weights(backbone, model_dir="./model_data"):
    import os
    from torch.hub import load_state_dict_from_url

    download_urls = {
        'mobilenet': 'https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar',
        'xception': 'https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/xception_pytorch_imagenet.pth',
    }
    url = download_urls[backbone]

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        :param patience: 在早停前可以容忍多少个epoch没有改进
        :param verbose: 如果为True，则为每次提升时输出一条消息
        :param delta: 重要改进的最小变化
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, save_dir):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_dir)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'Early stopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_dir)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, save_dir):
        if self.verbose:
            print(f'validation loss decrease ({self.val_loss_min:.6f} --> {val_loss:.6f}).  saving model ...')
        # 保存模型的代码
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
        self.val_loss_min = val_loss
