import os, sys
import cv2
import numpy as np
import random
import glob
from PIL import Image
import os.path as osp
import pandas as pd

patch_all = "E:\\newTumorPatcheDetection\\all_patches\\"

patch_dir = os.listdir(patch_all)
dst = "E:\\newTumorPatcheDetection\\divide\\"

for item in patch_dir:
    filename, e = os.path.splitext(item)
    im = cv2.imread(patch_all+ item)
    imgwidth = im.shape[0]
    imgheight = im.shape[1]

    y1 = 0
    M = 256
    N = 256
    os.makedirs(dst+ filename, exist_ok=True)
    for x in range(0, imgwidth, M):
        for y in range(0, imgheight, N):
            x1 = x + M
            y1 = y + N
            tiles = im[x:x + M, y:y + N]
            if(tiles.shape[0] == 256 and tiles.shape[1] == 256):
                print(y1)
                cv2.imwrite(dst+ "/"+ filename + "/" +filename+"_"+ str(x)+"_" + str(y) + ".png", tiles)
