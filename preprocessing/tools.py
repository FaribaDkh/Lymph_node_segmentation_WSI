import pandas as pd
import os
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
import cv2
import random as rng
i = 0
df = pd.DataFrame({"patient_name": [], "number_of_patches": []})
dst = "E:\\dst\\"
all_image = "E:\\dst_image\\"
dst_name_dir = os.listdir(dst)
patient_name = ""
for item in dst_name_dir:
    dst_name, e = os.path.splitext(item)
    images_dir = os.listdir(dst + dst_name)
    for item1 in images_dir:
        images_name, e = os.path.splitext(item1)
        dst_name_dir = os.listdir(dst)
        parts = dst_name.split('_')
        arr = np.array(parts)
        # print(arr.shape)
        if (arr.shape[0] > 1):

            img = cv2.imread(dst +dst_name+"\\"+ images_name+".png", 1)
            cv2.imwrite(all_image +"images\\"+ images_name+".png", img)

        else:
            patient_name = item
            img = cv2.imread(dst +dst_name+"\\"+ images_name+".png", 1)
            cv2.imwrite(all_image+"masks\\" + images_name+ ".png", img)
            i += 1
    df = df.append({"patient_name": patient_name, "number_of_patches": i}, ignore_index=True)
    i = 0

df.to_csv('dataset_number_of_patches_per_patient.csv')