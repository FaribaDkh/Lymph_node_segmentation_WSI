import os, sys
import cv2
import numpy as np
import random
import glob
from PIL import Image
import os.path as osp
import pandas as pd
import matplotlib.pyplot as plt
import csv

allPatches = "E:\\camelyon16\\new_dataset\\splitted_dataset\\test\\"
allmask = "E:\\camelyon16\\new_dataset\\splitted_dataset\\test_mask_1\\"
# result ="E:\\camelyon16\\50k_dataset\\dataset_splitted\\output_50k_wbn\\visualize\\"
# result ="E:\\camelyon16\\50k_dataset\\dataset_splitted\\TransUnet\\"
result ="E:\\camelyon16\\new_dataset\\splitted_dataset\\results\\wbn_mini\\visualize\\"
dst_path = 'E:\\camelyon16\\new_dataset\\splitted_dataset\\results\\'
type = "Kmeans_Five_layers_with_BN"
with open("E:\\camelyon16\\new_dataset\\splitted_dataset\\" + type + '.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    i = 0
    for row in csv_reader:
        if i > 1 :
            file_name = row[1]
            im = cv2.imread(allPatches + file_name + '.png', 1)
            im_mask = cv2.imread(allmask + file_name + '.png', 1)
            im_res = cv2.imread(result + file_name + '.png', 1)
            # im_res.astype(np.int32)
            im.astype(np.int32)
            # im_rgb = cv2.cvtColor(im_res, cv2.COLOR_BGR2GRAY)
            im_res = (im_res*255).astype(np.uint8)
            im_res_1 = (im_res*255).astype(np.uint8)
            im_res[:,:, 1] = 0
            im_res[:,:, 2] = 0

            fig, ax = plt.subplots(1, 3, figsize=(10, 9))
            dst = cv2.addWeighted(im, 0.5, im_res, 0.5, 0.0, dtype=cv2.CV_32F)
            dst = dst.astype(np.uint8)
            im_mask = im_mask.astype(np.uint8)
            ax[0].imshow(im)
            ax[1].imshow(im_mask*255)
            ax[2].imshow(im_res_1)
            ax[0].title.set_text(file_name)
            ax[1].title.set_text('mask')
            ax[2].title.set_text('Result')
            plt.savefig(dst_path+type+"\\"+ file_name + ".png", bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            plt.clf()
            plt.cla()
        i += 1
