import os, sys
import cv2
import numpy as np
import random
import glob
from PIL import Image
import os.path as osp
import pandas as pd
# E:\camelyon16\level2_1024\tumor_slide_mask
# E:\camelyon16\dataset_for_training\cam2017\level1\patient_004_node_4\\masks_1

# allPatches = "E:\\camelyon16\\dataset_for_training\\10k_dataset\\imgs\\tumor\\"
# maskDst = "E:\\camelyon16\\dataset_for_training\\10k_dataset\\imgs\\10k\\tumor_mask\\"
allPatches = "E:\\camelyon16\\dataset_for_training\\10k_dataset\\test\\"
allMaskPath = "E:\\camelyon16\\dataset_for_training\\10k_dataset\\test\\"

maskDst = "E:\\camelyon16\\dataset_for_training\\10k_dataset\\test\\"
# maskDst_1 = "E:\\camelyon16\\dataset_for_training\\10k_dataset\\10k_level2\\tumor_mask_1\\"
tumorDst =  "E:\\camelyon16\\dataset_for_training\\10k_dataset\\test\\"
# tumorDst_1 = "E:\\camelyon16\\level_2\\n_t_mask_1\\"
allMaskPathdirs = os.listdir(allMaskPath)
allPatchesdirs = os.listdir(allPatches)
maskdstDir = os.listdir(maskDst)
images = glob.glob(allPatches + "*.png")

folders= 'E:\\camelyon16\\dataset_for_training\\10k_dataset\\MIL\\'

dst = "E:\\camelyon16\\MIL_sub\\"
all_folder_dir = os.listdir(folders)

def folder_creation():
    df = pd.DataFrame({"slide_name": [], "slide_sub_folder": [], "label": []})
    bag_size = 100
    for item in all_folder_dir:
        filenameMask, e = os.path.splitext(item)
        path_folder = os.listdir(folders+ "/"+ filenameMask+"/")
        number_of_tiles = len(path_folder)
        if number_of_tiles < 100:
            number_of_sub_folder = 1
        else:
            number_of_sub_folder = number_of_tiles//bag_size
        patches = folders + "/" + filenameMask + "/"
        patches_dir = os.listdir(patches)
        subfolder = 0
        folder_name_dst = dst + filenameMask + "_" + str(subfolder)
        parts = filenameMask.split('_')
        # while(subfolder<number_of_sub_folder):
        #     try:
        #         os.mkdir(dst + filenameMask + "_" + str(subfolder))
        #     except OSError:
        #         print("Creation of the directory %s failed" % folder_name_dst)
        #     else:
        #         print("Successfully created the directory %s " % folder_name_dst)
        #     subfolder+=1
        subfolder = 1
        for patch in patches_dir:
            im = cv2.imread(patches + patch, 1)
            folder_name_dst = dst + filenameMask + "_" + str(subfolder-1)
            if len(os.listdir(folder_name_dst)) < bag_size:
                cv2.imwrite(dst + filenameMask + "_" + str(subfolder-1)+'/'+ patch, im)
            elif(subfolder<number_of_sub_folder):
                subfolder+=1




def create_df():
    folders = "E:\\camelyon16\\MIL_sub\\"
    df = pd.DataFrame({"slide_name": [], "slide_sub_folder": [], "label": []})
    folders_dir = os.listdir(folders)
    for item in folders_dir:
        filename, e = os.path.splitext(item)
        parts = filename.split('_')
        if (parts[0] == "tumor"):
            df = df.append(
                {"slide_name": filename, "case_id": filename,
                 "label": 1}, ignore_index=True)
        elif (parts[0] == "normal"):
            df = df.append(
                {"slide_name": filename, "case_id": filename,
                 "label": 0}, ignore_index=True)
    df.to_csv("df_sub.csv")


def number_of_patch_in_each_folder():
    folders = "E:\\camelyon16\\MIL_sub\\"
    folders_dir = os.listdir(folders)
    for item in folders_dir:
        item_dir = os.listdir("E:\\camelyon16\\MIL_sub\\"+ item)
        num = len(item_dir)
        if num<60:
            print(item +": " +str(num))



if __name__ == '__main__':
    # folder_creation()
    create_df()
    number_of_patch_in_each_folder()