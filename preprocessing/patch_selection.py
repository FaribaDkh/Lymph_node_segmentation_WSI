import os, sys
import cv2
import numpy as np
import random
import glob
from PIL import Image
import os.path as osp
import pandas as pd

patch_all = "E:\\camelyon16\\new_dataset\\all_patches\\"
dst = "E:\\camelyon16\\new_dataset\\masks\\"
dst_1 = "E:\\camelyon16\\new_dataset\\masks_1\\"
# tumorDst_1 = "E:\\camelyon16\\level_2\\n_t_mask_1\\"
allPatchesdirs = os.listdir(patch_all)
mask = "E:\\camelyon16\\50k_dataset\\dataset\\all_patches_mask\\"
mask_1 = "E:\\camelyon16\\50k_dataset\\dataset\\all_patches_mask_1\\"
# images = glob.glob(allPatches + "*.png")
#
# images.sort()

def find_specific_image_from_folder():
    for item in allPatchesdirs:
        filenameMask, e = os.path.splitext(item)
        parts = filenameMask.split('_')
        mask_img = cv2.imread(mask + filenameMask+".png",0)
        mask_1_img = cv2.imread(mask_1 + filenameMask+".png",0)
        cv2.imwrite(dst + filenameMask + ".png", mask_img)
        cv2.imwrite(dst_1 + filenameMask + ".png", mask_1_img)
def tottaly_white(arr, threshold= 1):
    tot = np.float(np.sum(arr))

    if (tot / arr.size == (threshold)):
       return True
    else:
       return False

# def find_specific_image_from_folder():
#     for item in allMaskPathdirs:
#         filenameMask, e = os.path.splitext(item)
#         parts = filenameMask.split('_')
#         image = cv2.imread(allMaskPath + filenameMask+".png",1)
#         cv2.imwrite(maskDst + parts[0] + "_" + parts[1] + "_" + parts[2] + "_" + parts[3] + ".png", image)
#         # cv2.imwrite(maskDst + filenameMask + ".png", image)

def random_selection_mask_img_pair():
    for tumor_slide in allPatchesdirs:
        filename, e = os.path.splitext(tumor_slide)
        file_name_list = filename.split('_')
        lens_of_list = len(file_name_list)
        if lens_of_list == 2:
            images = glob.glob(patch_all + "\\"+filename + "\\" + "*.png")
            images_dir = os.listdir(patch_all + "\\" + filename + "\\")
            len_of_image = len(images)
            if len_of_image < 300:
                for item in images_dir:
                    image_name, e = os.path.splitext(item)
                    image = cv2.imread(patch_all + "\\"+filename + "\\"+image_name + ".png", 1)
                    image_mask = cv2.imread(patch_all + "\\"+filename+"_mask" + "\\"+ image_name + ".png", 0)
                    image_mask_1 = cv2.imread(patch_all + "\\"+filename+"_mask_1" + "\\"+ image_name + ".png", 1)
                    cv2.imwrite(dst+"\\" + filename + "_" + image_name + ".png", image)
                    cv2.imwrite(dst+"_mask\\" + filename+"_"+image_name+"_mask"+ ".png", image_mask)
                    cv2.imwrite(dst+"_mask_1\\" + filename+"_"+image_name + "_mask_1" + ".png", image_mask_1)
            else:
                images = glob.glob(patch_all+ "\\"+filename + "\\" + "*.png")
                for i in range(400):
                    key = random.choice(list(images))
                    image_name = osp.splitext(osp.basename(key))[0]
                    image = cv2.imread(patch_all + "\\" + filename + "\\" + image_name + ".png", 1)
                    image_mask = cv2.imread(patch_all + "\\" + filename + "_mask" + "\\" + image_name + ".png", 0)
                    image_mask_1 = cv2.imread(patch_all + "\\" + filename + "_mask_1" + "\\" + image_name + ".png",1)
                    cv2.imwrite(dst + "\\" + filename+ "_" + image_name + ".png", image)
                    cv2.imwrite(dst + "_mask\\" + filename+ "_" + image_name + "_mask" + ".png", image_mask)
                    cv2.imwrite(dst + "_mask_1\\" + filename+ "_" + image_name + "_mask_1" + ".png", image_mask_1)


    # images = glob.glob(allPatches + "*.png")
    # for i in range(1):
    #     key = random.choice(list(images))
    #     filename = osp.splitext(osp.basename(key))[0]
    #     if os.path.isfile(allMaskPath + filename + "_mask.png"):
    #         im_mask = Image.open(allMaskPath + filename + "_mask.png")
    #         image = cv2.imread(allMaskPath + filename + "_mask.png", 1)
    #         if (tottaly_white(image) == False):
    #             image = image / 255
    #         parts = filename.split('_')
    #         im = Image.open(allPatches + filename + ".png")
    #         im.save(tumorDst + filename + ".png")
    #         # image.save(maskDst + parts[0] + "_" + parts[1] + "_" + parts[2] + "_" + parts[3] + ".png")
    #         cv2.imwrite(maskDst + parts[0] + "_" + parts[1] + "_" + parts[2] + "_" + parts[3] + ".png", image)
    #         # cv2.imwrite(maskDst_1 + filename + ".png", image)
    #         print(key)

if __name__ == '__main__':
    find_specific_image_from_folder()