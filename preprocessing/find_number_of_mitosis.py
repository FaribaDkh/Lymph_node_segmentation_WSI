import os, sys
import cv2
import numpy as np
import random
import glob
from PIL import Image
import os.path as osp
import pandas as pd
# allPatches = "E:\\camelyon16\\dataset_for_training\\10k_dataset\\imgs\\tumor\\"
# maskDst = "E:\\camelyon16\\dataset_for_training\\10k_dataset\\imgs\\10k\\tumor_mask\\"
allPatches = "C:\\Users\\fdamband\\Desktop\\MIDL\\ICPR2012\\masks\\"
img_patch = "C:\\Users\\fdamband\\Desktop\\MIDL\\ICPR2012\\patch\\"
dst = "C:\\Users\\fdamband\\Desktop\\MIDL\\"

# tumorDst_1 = "E:\\camelyon16\\level_2\\n_t_mask_1\\"
allPatchesdirs = os.listdir(allPatches)

images = glob.glob(allPatches + "*.png")

images.sort()
count_mitosis = 0
kernel = np.ones((3,3), np.uint8)
def remove_small_particles(greenPixels):
    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(greenPixels, connectivity=8)
    # connectedComponentswithStats yields every seperated component with information on each of them, such as size
    # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1];
    nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 50

    # your answer image
    img2 = np.zeros((output.shape))
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    kernel = np.ones((2, 2), np.uint8)
    img2 = cv2.dilate(img2, kernel, iterations=1)
    img2 = cv2.dilate(img2, kernel, iterations=1)
    img2 = cv2.dilate(img2, kernel, iterations=1)
    img2 = cv2.dilate(img2, kernel, iterations=1)
    return img2
for item in allPatchesdirs:
    file_name, e = os.path.splitext(item)
    mask = cv2.imread(allPatches+ item,1)
    img = cv2.imread(img_patch+ file_name+ "_original.png",1)
    # img = remove_small_particles(img)
    gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
    # dst_img = cv2.addWeighted(img, 0.7, gradient, 0.3, 1)
    # print(dst_img.dtype)
    cv2.imwrite(dst + item, gradient)

# for item in allPatchesdirs:
#     file_name,e = os.path.split(item)
#     img = cv2.imread(allPatches+ item,1)
#
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     mask_1 = cv2.inRange(hsv, (36, 25, 25), (70, 255, 255))
#     mask_2 = cv2.inRange(hsv, (0, 50, 20), (5, 255, 255))
#     mask_3 = cv2.inRange(hsv, (175, 50, 20), (180, 255, 255))
#     greenPixels = cv2.merge((mask_1, mask_1, mask_1))
#     greenPixels = cv2.cvtColor(greenPixels, cv2.COLOR_HSV2BGR)
#     green_pixels = greenPixels[:,:,1]/255.0
#     greenPixels = cv2.cvtColor(greenPixels, cv2.COLOR_BGR2GRAY)
#     cent, _ = cv2.findContours(greenPixels, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     count_mitosis += len(cent)
# print(count_mitosis)

