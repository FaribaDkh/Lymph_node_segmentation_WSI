import numpy as np
import random
import glob
# import imageio
from PIL import Image
import sys
import os
import os.path as osp
import cv2
# base_path = "E:\\camelyon16\\level2\\"
# dst_imgs_path = "E:\\camelyon16\\dataset_for_training\\level_2\\dataset_all_tumor"
# E:\camelyon16\dataset_for_training\level_2\dataset_all_tumor
dst_imgs_path = "E:\\camelyon16\\new_dataset\\splitted_dataset\\"

# dst_imgs_path = "E:\\camelyon16\\dataset_for_training\\level_1\\dataset\\more_than_70"
# images_path = "E:\\camelyon16\\dataset_for_training\\level_1\\imgs_70\\"
# segs_path = "E:\\camelyon16\\dataset_for_training\\level_1\\masks_70\\"
# images_path = "E:\\camelyon16\\level2_512\\dataset\\imgs\\"

# # segs_path = "E:\\camelyon16\\level2_512\\ntmask\\masks\\"
segs_path = "E:\\camelyon16\\new_dataset\\masks\\"
segs_path_1 = "E:\\camelyon16\\new_dataset\\masks_1\\"
images_path = "E:\\camelyon16\\new_dataset\\all_patches\\"

images = glob.glob(images_path + "*.png")

images.sort()

dataset_size = len(images)
indices = list(range(dataset_size))
split_1 = int(np.floor(0.7 * dataset_size))
split_2 = int(np.floor(0.9 * dataset_size))

# split_1 = int(5000)
# split_2 = int(50000)
# split = int(np.floor(val_percent * dataset_size))
##as validation set is splitted in the training code I did not split val
np.random.shuffle(indices)
## use split 1 for separating train and val
train_indices = indices[:split_1]
# train_indices = indices[:split_2]
test_indices = indices[split_2:]
val_indices = indices[split_1:split_2]
def is_sorta_black(arr, threshold= 0.1):
    tot = np.float(np.sum(arr))
    # print (tot / arr.size)
    # print (tot)
    # print (arr.size)

    # if (tot / arr.size > (threshold)|((tot/arr.size) == 1)):
    if (tot / arr.size > (threshold)):
    # if (tot / arr.size != 255):
    #    print ("is not black" )
       return True
    else:
       # print ("is kinda black")
       return False
def tottaly_white(arr, threshold= 1):
    tot = np.float(np.sum(arr))
    # print (tot / arr.size)
    # print (tot)
    # print (arr.size)

    # if (tot / arr.size > (threshold)|((tot/arr.size) == 1)):
    if (tot / arr.size == (threshold)):
    # if (tot / arr.size != 255):
    #    print ("is not black" )
       return True
    else:
       # print ("is kinda black")
       return False

# Detecting masks with tumor area

for i in range(len(train_indices)):
    im = Image.open(images[train_indices[i]]).convert("RGB")
    filename = osp.splitext(osp.basename(images[train_indices[i]]))[0]
    # im_mask = Image.open(segs_path + filename + "_mask" + ".png")
    im_mask = cv2.imread(segs_path + filename + ".png",1)
    im_mask_1 = cv2.imread(segs_path_1 + filename + ".png",1)
    Image.open(images[train_indices[i]]).convert("RGB")
    # if (tottaly_white(im_mask) == False):
    #     im_mask = im_mask / 255
    # if (is_sorta_black(im_mask)):  # Get the RGBA Value of the a pixel of an image
    #     im_mask = im_mask * 255
    cv2.imwrite(dst_imgs_path + "\\train_mask\\" + filename + ".png", im_mask)
    cv2.imwrite(dst_imgs_path + "\\train_mask_1\\" + filename + ".png", im_mask_1)
    im.save(dst_imgs_path + "\\train\\" + filename + ".png")
    # im_mask.save(train_mask_path + filename + "_mask" + ".png")
#
for i in range(len(val_indices)):
    im = Image.open(images[val_indices[i]])
    filename = osp.splitext(osp.basename(images[val_indices[i]]))[0]
    im_mask = Image.open(segs_path + filename + ".png")
    im_mask_1 = Image.open(segs_path_1 + filename + ".png")
    im.save(dst_imgs_path + "\\val\\"+filename + ".png")
    im_mask.save(dst_imgs_path + "\\val_mask\\" + filename + ".png")
    im_mask_1.save(dst_imgs_path + "\\val_mask_1\\" + filename + ".png")

for i in range(len(test_indices)):
    im = Image.open(images[test_indices[i]])
    filename = osp.splitext(osp.basename(images[test_indices[i]]))[0]
    im_mask = Image.open(segs_path + filename + ".png")
    im_mask_1 = Image.open(segs_path + filename + ".png")
    im.save(dst_imgs_path + "\\test\\"+filename + ".png")
    im_mask.save(dst_imgs_path + "\\test_mask\\" + filename + ".png")
    im_mask_1.save(dst_imgs_path + "\\test_mask_1\\" + filename + ".png")
# im_mask = Image.open(segs_path + filename + "_mask" + ".png")
# test_indices = indices[split_2:]
# val_indices = indices[split_1:split_2]
# segmentations = glob.glob(segs_path + "*.png")
# segmentations.sort()
# for i in range(1):
#     key = random.choice(list(images))
#     im = Image.open(key)
#     filename = osp.splitext(osp.basename(key))[0]
#     im_mask = Image.open(segs_path + filename + "_mask" + ".png")
#     im = Image.open(images_path + filename + ".png")
#     im.save(dataset_imgs_path + filename + ".png")
#     im_mask.save(dataset_mask_path + filename + "_mask" + ".png")
#     print(key)



# def copy_mask_with_same_tumor_name():
#     for item2 in allPatchesdirs:
#         for item in allMaskPathdirs:
#
#             filenameMask, e = os.path.splitext(item)
#             fileNameTumor, e1 = os.path.splitext(item2)
#             if (filenameMask == fileNameTumor+"_mask"):
#                 im = Image.open(allMaskPath + filenameMask+".png")
#                 im.save(tumorDst + item)