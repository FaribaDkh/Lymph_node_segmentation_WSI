from PIL import Image
import os, sys
import numpy as np
import cv2

masks_path = "E:\\camelyon16\\dataset_for_training\\level_2\\dataset\\train_mask\\"
masks_dirs = os.listdir(masks_path)
count_normal_pixel = 0
count_tumor_pixel = 0
for item in masks_dirs:
    if os.path.isfile(masks_path + item):
        img = cv2.imread(masks_path+ item, 0)
        im_mask = np.asarray(Image.open(masks_path + item))
        p = img.shape
        rows, cols = img.shape
        for i in range(rows):
            for j in range(cols):
                k = img[i, j]
                # print(k)
                if(k== 255):
                    count_tumor_pixel = count_tumor_pixel+ 1
                else:
                    count_normal_pixel = count_normal_pixel+ 1
    print("count_tumor_pixel: " + str(count_tumor_pixel))
    print("count_normal_pixel: " + str(count_normal_pixel))
print("count_normal_pixel: " + str(count_normal_pixel))
print("count_tumor_pixel: " + str(count_tumor_pixel))