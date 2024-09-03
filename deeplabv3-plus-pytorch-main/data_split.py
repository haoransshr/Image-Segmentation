import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

num_images_to_sample = 0  # 0 to use all images in path
train_percent = 0.8
valid_percent = 0.1
test_percent = 0.1

path = './datasets'

if __name__ == "__main__":
    random.seed(0)
    print("Generate split list txt in ImageSets.")
    segfilepath = os.path.join(path, 'labels')
    saveBasePath = os.path.join(path, 'lists')

    print(segfilepath)
    print(saveBasePath)

    temp_seg = os.listdir(segfilepath)
    total_seg = []
    for seg in temp_seg:
        if seg.endswith(".png"):
            total_seg.append(seg)

    if num_images_to_sample > 0:
        sampled_seg = random.sample(total_seg, num_images_to_sample)
    else:
        sampled_seg = total_seg

    num = len(sampled_seg)
    indices = list(range(num))
    train_size = int(num * train_percent)
    val_size = int(num * valid_percent)
    test_size = num - train_size - val_size

    trainval_indices = random.sample(indices, train_size + val_size)
    train_indices = random.sample(trainval_indices, train_size)
    val_indices = list(set(trainval_indices) - set(train_indices))
    test_indices = list(set(indices) - set(trainval_indices))

    print("train size", train_size)
    print("val size:", val_size)
    print("test size:", test_size)
    ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
    ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
    fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')
    ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')

    for i in trainval_indices:
        name = sampled_seg[i][:-4] + '\n'
        ftrainval.write(name)
        if i in train_indices:
            ftrain.write(name)
        else:
            fval.write(name)

    for i in test_indices:
        name = sampled_seg[i][:-4] + '\n'
        ftest.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()
    print("Generate txt in ImageSets done.")

    print("Check datasets format, this may take a while.")
    classes_nums = np.zeros([256], np.int64)
    for i in tqdm(indices):
        name = sampled_seg[i]
        png_file_name = os.path.join(segfilepath, name)
        if not os.path.exists(png_file_name):
            raise ValueError("No image found %s, please check if it's exist." % (png_file_name))

        png = np.array(Image.open(png_file_name), np.uint8)
        if len(np.shape(png)) > 2:
            print("The image %s has the shape of %s, which format is invalid." % (name, str(np.shape(png))))

        classes_nums += np.bincount(np.reshape(png, [-1]), minlength=256)

    print("print the pixels' value and number")
    print('-' * 37)
    print("| %15s | %15s |" % ("Key", "Value"))
    print('-' * 37)
    for i in range(256):
        if classes_nums[i] > 0:
            print("| %15s | %15s |" % (str(i), str(classes_nums[i])))
            print('-' * 37)
