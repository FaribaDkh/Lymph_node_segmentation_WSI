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

dst = "E:\\dst\\"
center_path = "E:\\mitosis\\"
cvs_path = "C:\\Users\\fdamband\\Downloads\\mitoses_ground_truth\\"
center_name_dir = os.listdir(center_path)


def crop_HPF():
    for item1 in  center_name_dir:
        center, e = os.path.splitext(item1)
        patient = center_path + center +"\\"
        patient_name_dir = os.listdir(patient)
        for item in patient_name_dir:
            patient_name, e = os.path.splitext(item)
            try:
                os.mkdir(dst + patient_name)
            except OSError:
                print("Creation of the directory %s failed")
            else:
                print("Successfully created the directory %s ")
            try:
                os.mkdir(dst + patient_name+"_mask")
            except OSError:
                print("Creation of the directory %s failed")
            else:
                print("Successfully created the directory %s ")

            HPFz = patient + patient_name + "\\"
            HPFz_dir = os.listdir(HPFz)
            for item2 in HPFz_dir:
                HPFz_name, e = os.path.splitext(item2)
                #  HPF path is defined below
                HPFz_image = HPFz + HPFz_name
                HPFz_name_csv = cvs_path + patient_name + "\\" + HPFz_name
                im = Image.open(HPFz_image + ".tif")
                image = cv2.imread(HPFz_image + ".tif",1)
                image_mask = cv2.imread(HPFz_image + ".tif",1)
                # counting lines of each csv files which shows number of mitosis
                if os.path.isfile(HPFz_name_csv + ".csv"):
                    with open(HPFz_name_csv + ".csv", "r") as f:
                        reader = csv.reader(f, delimiter=",")
                        data = list(reader)
                        data_1 = [row for row in reader]
                        row_count = len(data)
                        dataCoord = pd.read_csv(HPFz_name_csv + ".csv", header=None)
                        locations = [(int(x[0]), int(x[1])) for x in dataCoord.values.tolist()]
                        for loc in locations:
                            cv2.rectangle(image_mask, (loc[1] - 20, loc[0] - 20), (loc[1] + 20, loc[0] + 20),
                                          (0, 255, 0), 2)
                        row = int (image.shape[0]/256)
                        for i in range(row+1):
                            for j in range(row+1):

                                start_coor_x = i*256
                                start_coor_y = j*256
                                cropped = image[start_coor_x:start_coor_x+256, start_coor_y:start_coor_y + 256]
                                cropped_mask = image_mask[start_coor_x:start_coor_x+256, start_coor_y:start_coor_y + 256]

                            # for loc in locations:
                            # print((loc[1] - 20, loc[0] - 20))
                            # print((loc[1] + 20, loc[0] + 20))
                                for loc in locations:
                                    if ((start_coor_x <= loc[0] <= start_coor_x+256)& (start_coor_y <= loc[1] <= start_coor_y+256)):
                                        # cropped_mask = cv2.circle(cropped, (int(data[i][0]), int(data[i][1])), 50, (0,255,0), 2)
                                        image_name = "Tupac_ROI_Training_" + "patient" + patient_name + "_HPF" + HPFz_name + "_mistos_X_" + str(start_coor_x)+"_Y_"+str(start_coor_y)
                                        # if (cropped.shape[0] ==256 & cropped.shape[1] == 256):
                                        cv2.imwrite(dst + patient_name + "\\" + image_name + ".png", cropped)
                                        cv2.imwrite(dst + patient_name+"_mask\\" + image_name + ".png", cropped_mask)

                            # box = (int(data[i][0]) + random_num_1, int(data[i][1]) + random_num_2, 256, 256)
                            # img2 = im.crop((data[i][0] + random_num_1 ,data[i][1] + random_num_2, 256, 256))
                            # img2 = im.crop(box)
                            # img2.save("img2.jpg")
if __name__ == '__main__':
    crop_HPF()
