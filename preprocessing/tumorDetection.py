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
allPatches = "E:\\camelyon16\\new_dataset\\splitted_dataset\\test_mask_1\\"
allMaskPath = "E:\\camelyon16\\dataset_for_training\\10k_dataset\\test\\"

maskDst = "E:\\camelyon16\\dataset_for_training\\10k_dataset\\test\\"
# maskDst_1 = "E:\\camelyon16\\dataset_for_training\\10k_dataset\\10k_level2\\tumor_mask_1\\"
tumorDst =  "E:\\camelyon16\\dataset_for_training\\10k_dataset\\test\\"
# tumorDst_1 = "E:\\camelyon16\\level_2\\n_t_mask_1\\"
allMaskPathdirs = os.listdir(allMaskPath)
allPatchesdirs = os.listdir(allPatches)
maskdstDir = os.listdir(maskDst)
images = glob.glob(allPatches + "*.png")
# Detecting masks with tumor area
maskDst = "E:\\camelyon16\\dataset_for_training\\10k_dataset\\dataset_1\\val_images\\val_masks\\val_1\\"
trDst = "E:\\camelyon16\\dataset_for_training\\10k_dataset\\dataset_1\\val_images\\val_frames\\val_1\\"
images.sort()
# maskdstDir_1 = os.listdir(tumorDst_1)

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
def tumor_mask():
    i = 0
    for item in allMaskPathdirs:
        if os.path.isfile(allMaskPath + item):
            filenameMask, e = os.path.splitext(item)
            # im = cv2.imread(allPatches + item, 1)
            im_mask = cv2.imread(allMaskPath + item, 0)
            im_mask = im_mask*255
            cv2.imwrite(allMaskPath + item, im_mask)
            # cv2.imwrite(trDst + str(i)+".png", im)
            # i+=1
def change_name():
    for item in allMaskPathdirs:
        if os.path.isfile(allMaskPath + item):
            filenameMask, e = os.path.splitext(item)
            im_mask = cv2.imread(allMaskPath + item, 0)
            im_mask = im_mask * 255
def tumor_mask_detection():
    for item in allMaskPathdirs:
        if os.path.isfile(allMaskPath + item):
            filenameMask, e = os.path.splitext(item)
            im = cv2.imread(allMaskPath + item,0)
            parts = filenameMask.split('_')
            # cv2.imwrite(maskDst + parts[0] + "_" + parts[1] + "_" + parts[2] + "_" + parts[3] + ".png", image)
            # pix = im.load()
            # [x,y]= im.size  # Get the width and hight of the image for iterating over
            if(tottaly_white(im) == False):
                im = im/255
            # if (is_sorta_black(im)): # Get the RGBA Value of the a pixel of an image
            #     im = im *255
            cv2.imwrite(maskDst + parts[0] + "_" + parts[1] + "_" + parts[2] + "_" + parts[3] + ".png", im)
            # cv2.imwrite(maskDst+item,im)
            # pix[x, y] = value  # Set the RGBA Value of the image (tuple)
            # im.save('alive_parrot.png')  # Save the modified pixels as .png

            # imResize = im.resize((256, 256), Image.ANTIALIAS)
            # imResize.convert('RGB').save(f+'.png', 'png', quality=80)
# Finding the same tumor image with the same mask name and copy to the specicif folder
def copy_tumor_with_same_mask_name():
    # searching by two for
    # You can use it with one 'for'. Searching with its name
    # random selection function is written with one
    for item in maskdstDir:
        for item2 in allPatchesdirs:

            filenameMask, e = os.path.splitext(item)
            fileNameTumor, e1 = os.path.splitext(item2)
            if (filenameMask == fileNameTumor+"_mask"):
                im = Image.open(allPatches + fileNameTumor+".png")
                im.save(tumorDst + item2)
def copy_mask_with_same_tumor_name():
    for item2 in allPatchesdirs:
        for item in allMaskPathdirs:

            filenameMask, e = os.path.splitext(item)
            fileNameTumor, e1 = os.path.splitext(item2)
            if (filenameMask == fileNameTumor+"_mask"):
                im = Image.open(allMaskPath + filenameMask+".png")
                im.save(tumorDst + item)
def copy_mask_with_same_tumor_name_2():
    for item in allPatchesdirs:
        filenameMask, e = os.path.splitext(item)
        parts = filenameMask.split('_')
        im = cv2.imread(allMaskPath + parts[0] + "_" + parts[1] + "_" + parts[2] + "_" + parts[3] + "_mask"+".png", 1)
        # if (tottaly_white(im) == False):
        #     im = im / 255

        # fileNameTumor, e1 = os.path.splitext(item2)
        # im = Image.open(allMaskPath + filenameMask+"_mask.png")
        cv2.imwrite(maskDst + parts[0] + "_" + parts[1] + "_" + parts[2] + "_" + parts[3] + ".png", im)
        # im.save(maskDst + item)
def create_empty_mask_for_patches():
    for item in allPatchesdirs:
        filenameMask, e = os.path.splitext(item)
        img = np.zeros([512, 512, 3], dtype=np.uint8)
        img.fill(0)  # or img[:] = 255
        # im = Image.open(allMaskPath + filenameMask+".png")
        cv2.imwrite(maskDst + filenameMask +".png",img)
        # img.save(tumorDst_1 + filenameMask+"mask"+".png")
def convert_four_channels_to_three():
    for item in allPatchesdirs:
        if os.path.isfile(allPatches + item):
            im = cv2.imread(allPatches + item, 1)
            # rgb_binary = cv2.inRange(im, 100, 255)
            img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(allPatches + item, img)
            # pix = im.load()
            # [x,y]= im.size  # Get the width and hight of the image for iterating over
def crop_from_center_image():
    image_size = 256
    all_patches = "E:\\camelyon16\\dataset_for_training\\10k_dataset\\test_cam2017\\level2\\imgs\\"
    allPatchesdirs = os.listdir(all_patches)
    all_masks = "E:\\camelyon16\\dataset_for_training\\10k_dataset\\test_cam2017\\level2\\masks\\"
    all_masks_1 = "E:\\camelyon16\\dataset_for_training\\10k_dataset\\test_cam2017\\level2\\masks_1\\"
    dst = "E:\\camelyon16\\dataset_for_training\\10k_dataset\\test_cam2017\\"
    for item in allPatchesdirs:
        if os.path.isfile(all_patches + item):
            filename, e = os.path.splitext(item)
            im = Image.open(all_patches + item)
            im_mask = Image.open(all_masks + item)
            im_mask_1 = Image.open(all_masks_1 + item)
            width, height = im.size
            # # Level2_256
            im_level2_256 = im.resize((512, 512))
            im_mask_level2_256 = im_mask.resize((512, 512))
            im_mask_level2_256_1 = im_mask_1.resize((512, 512))
            left = (width - 256) / 2
            top = (height - 256) / 2
            right = (width + 256) / 2
            bottom = (height + 256) / 2
            im_LEVEL2_256 = im_level2_256.crop((left, top, right, bottom))
            im_mask_LEVEL2_256 = im_mask_level2_256.crop((left, top, right, bottom))
            im_mask_LEVEL2_1_256 = im_mask_level2_256_1.crop((left, top, right, bottom))
            im_LEVEL2_256.save(dst + "level2_256\\imgs\\" + filename + ".png")
            im_mask_LEVEL2_256.save(dst + "level2_256\\masks\\" + filename + ".png")
            im_mask_LEVEL2_1_256.save(dst + "level2_256\\masks_1\\" + filename + ".png")
            # # Level 0
            # left = (width - 128) / 2
            # top = (height - 128) / 2
            # right = (width + 128) / 2
            # bottom = (height + 128) / 2
            # im_LEVEL0 = im.crop((left, top, right, bottom))
            # im_mask_LEVEL0 = im_mask.crop((left, top, right, bottom))
            # im_mask_LEVEL0_1 = im_mask_1.crop((left, top, right, bottom))
            # im_LEVEL0 = im_LEVEL0.resize((256, 256))
            # im_mask_LEVEL0 = im_mask_LEVEL0.resize((256, 256))
            # im_mask_LEVEL0_1 = im_mask_LEVEL0_1.resize((256, 256))
            # im_LEVEL0.save(dst + "level0\\imgs\\" + filename + ".png")
            # im_mask_LEVEL0.save(dst + "level0\\masks\\" + filename + ".png")
            # im_mask_LEVEL0_1.save(dst + "level0\\masks_1\\" + filename + ".png")
            # #LEVEL1
            # # Get dimensions
            # left = (width - 256) / 2
            # top = (height - 256) / 2
            # right = (width + 256) / 2
            # bottom = (height + 256) / 2
            # im_LEVEL1 = im.crop((left, top, right, bottom))
            # im_mask_LEVEL1 = im_mask.crop((left, top, right, bottom))
            # im_mask_LEVEL1_1 = im_mask_1.crop((left, top, right, bottom))
            # im_LEVEL1.save(dst + "level1\\imgs\\" + filename + ".png")
            # im_mask_LEVEL1.save(dst + "level1\\masks\\" + filename + ".png")
            # im_mask_LEVEL1_1.save(dst + "level1\\masks_1\\" + filename + ".png")
            # # Level2
            # im_level2 = im.resize((512, 512))
            # im_mask_level2 = im_mask.resize((512, 512))
            # im_mask_level2_1 = im_mask.resize((512, 512))
            # im_level2.save(dst + "level2\\imgs\\" + filename + ".png")
            # im_mask_level2.save(dst + "level2\\masks\\" + filename + ".png")
            # im_mask_level2_1.save(dst + "level2\\masks_1\\" + filename + ".png")
def crop():
    dataset = "E:\\camelyon16\\dataset_for_training\\10k_dataset\\10k_level2\\dataset\\"
    dst = "E:\\camelyon16\\dataset_for_training\\10k_dataset\\10k_level2_256\\"
    Patchesdirs = os.listdir(dataset+"val\\")
    for item in Patchesdirs:
        if os.path.isfile(dataset+"val\\" + item):
            filename, e = os.path.splitext(item)
            im = Image.open(dataset + "val\\" + item)
            im_mask = Image.open(dataset + "val_mask\\"+ item)
            width, height = im.size
            left = (width - 256) / 2
            top = (height - 256) / 2
            right = (width + 256) / 2
            bottom = (height + 256) / 2
            im = im.crop((left, top, right, bottom))
            im_mask = im_mask.crop((left, top, right, bottom))
            im.save(dst + "val\\" + filename + ".png")
            im_mask.save(dst + "val_mask\\" + filename + ".png")
def resize():
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    allMaskPath =  "E:\\camelyon16\\dataset_for_training\\10k_dataset\\test_cam2017\\level0\\imgs\\"
    allMaskPath_1 =  "E:\\camelyon16\\dataset_for_training\\10k_dataset\\test_cam2017\\level0\\imgs_224\\"
    patches_dir = os.listdir(allMaskPath)
    for item in patches_dir:
        filenameMask, e = os.path.splitext(item)
        image = Image.open(allMaskPath + filenameMask+".png")
        new_image = image.resize((224, 224))
        # new_image = new_image / 255
        new_image.save(allMaskPath_1 + filenameMask+".png")
def mask_division():

    for item in allMaskPathdirs:

        filenameMask, e = os.path.splitext(item)
        parts = filenameMask.split('_')
        image = cv2.imread(allMaskPath + filenameMask + ".png", 1)
        if (tottaly_white(image) == False):
            image = image / 255
        # image = image * 255
        # cv2.imwrite(maskDst + parts[0]+"_"+parts[1]+"_"+parts[2]+"_"+parts[3]+".png", image)
        cv2.imwrite(allMaskPath + filenameMask+".png", image)
        # cv2.imwrite(maskDst +filenameMask+".png", image)
def find_specific_image_from_folder():
    for item in allMaskPathdirs:
        filenameMask, e = os.path.splitext(item)
        parts = filenameMask.split('_')
        image = cv2.imread(allMaskPath + filenameMask+".png",1)
        cv2.imwrite(maskDst + parts[0] + "_" + parts[1] + "_" + parts[2] + "_" + parts[3] + ".png", image)
        # cv2.imwrite(maskDst + filenameMask + ".png", image)
def random_selection_mask_img_pair():
    allPatches = "E:\\camelyon16\\dataset_for_training\\level_1\\n_t\\"
    allMaskPath = "E:\\camelyon16\\dataset_for_training\\level_1\\n_t_mask\\"
    # maskDst = "E:\\camelyon16\\dataset_for_training\\10k_dataset\\10k_level2\\patches\\normal_tumor_mask\\"
    maskDst = "E:\\camelyon16\\dataset_for_training\\10k_dataset\\10k_level0_balanced\\patches\\normal_tumor_mask\\"
    # maskDst_1 = "E:\\camelyon16\\dataset_for_training\\10k_dataset\\10k_level2\\tumor_mask_1\\"
    tumorDst = "E:\\camelyon16\\dataset_for_training\\10k_dataset\\10k_level0_balanced\\patches\\normal_tumor\\"
    images = glob.glob(allPatches + "*.png")
    for i in range(1):
        key = random.choice(list(images))
        filename = osp.splitext(osp.basename(key))[0]
        if os.path.isfile(allMaskPath + filename + "_mask.png"):
            im_mask = Image.open(allMaskPath + filename + "_mask.png")
            image = cv2.imread(allMaskPath + filename + "_mask.png", 1)
            if (tottaly_white(image) == False):
                image = image / 255
            parts = filename.split('_')
            im = Image.open(allPatches + filename + ".png")
            im.save(tumorDst + filename + ".png")
            # image.save(maskDst + parts[0] + "_" + parts[1] + "_" + parts[2] + "_" + parts[3] + ".png")
            cv2.imwrite(maskDst + parts[0] + "_" + parts[1] + "_" + parts[2] + "_" + parts[3] + ".png", image)
            # cv2.imwrite(maskDst_1 + filename + ".png", image)
            print(key)
def is_sorta_tumor(arr, threshold=0.1):
    tot = np.float(np.sum(arr))
    print (tot)
    print(tot/arr.size )
    if tot/arr.size <(threshold):
       # print ("is not black" )
       return False
    else:
       # print ("is kinda black")
       return True
def get_confusion_matrix_intersection_mats(groundtruth, predicted):
    """ Returns dict of 4 boolean numpy arrays with True at TP, FP, FN, TN
    """

    confusion_matrix_arrs = {}

    groundtruth_inverse = np.logical_not(groundtruth)
    predicted_inverse = np.logical_not(predicted)

    confusion_matrix_arrs['tp'] = np.logical_and(groundtruth, predicted)
    tp = np.count_nonzero(confusion_matrix_arrs['tp'])
    confusion_matrix_arrs['tn'] = np.logical_and(groundtruth_inverse, predicted_inverse)
    tn = np.count_nonzero(confusion_matrix_arrs['tn'])
    confusion_matrix_arrs['fp'] = np.logical_and(groundtruth_inverse, predicted)
    fp = np.count_nonzero(confusion_matrix_arrs['fp'])
    confusion_matrix_arrs['fn'] = np.logical_and(groundtruth, predicted_inverse)
    fn = np.count_nonzero(confusion_matrix_arrs['fn'])
    if(is_sorta_tumor(groundtruth) & is_sorta_tumor(predicted)):
        dsc = (2 * tp) / (2 * tp + fp + fn)
        precision = tp / (tp + fn)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
    return tp, tn, fp, fn, dsc, f1
def cofusion_matrix_two_folder():
    for item in allMaskPathdirs:
        filenameMask, e = os.path.splitext(item)
        gt = cv2.imread("E:\\camelyon16\\dataset_for_training\\level_2\\new_dataset\\test_unet\\mask\\" + filenameMask + ".png", 0)
        pr = cv2.imread("E:\\camelyon16\\dataset_for_training\\level_2\\new_dataset\\test_unet\\output\\" + filenameMask + ".png", 0)
        tp, tn, fp, fn, dsc, f1 = get_confusion_matrix_intersection_mats(gt, pr)
        print(tp)
        print(tn)
        print(fp)
        print(fn)
        print(dsc)
        print(f1)
def for_loop():
    for n in list(range(1, 10)) + list(range(10, 60, 10)):
        n = float(n)
        n = float(n/ 1000)
        print(n)
    # do something
    # for i in np.arange(0.0, 0.01,0.001) + np.arange(0.10,0.05,0.01):
    #     print(i)
def labeling():

    allPatches = "E:\\camelyon16\\dataset_for_training\\10k_dataset\\test\\"
    allMasks = "E:\\camelyon16\\dataset_for_training\\10k_dataset\\dataset\\test_images\\test_masks\\test\\"
    path = "E:\\camelyon16\\dataset_for_training\\10k_dataset\\"
    allPatchesdirs = os.listdir(allPatches)
    label = pd.DataFrame({"slide_name": [], "case_id": [], "label": []})
    for item in allPatchesdirs:
        filename, e = os.path.splitext(item)
        im_mask = cv2.imread(allMasks+"/" + filename + ".png",0)
        # image = cv2.imread(allPatches + filename + ".png", 1)
        if (cv2.countNonZero(im_mask) > 1000):
            label = label.append({"slide_name": filename, "case_id":filename, "label": 1}, ignore_index=True)
        else:
            label = label.append({"slide_name": filename, "case_id": filename, "label": 0}, ignore_index=True)
        label.to_csv(path + 'test_label.csv')
def search_patch_in_folder():
    patches = "E:\\camelyon16\\50k_dataset\\dataset_splitted\\test\\"
    source_path = "E:\\camelyon16\\50k_dataset\\dataset_splitted\\"
    check_dir = "E:\\camelyon16\\new_dataset\\all_patches\\"
    dst = 'E:\\camelyon16\\new_dataset\\'
    patch_dir = os.listdir(patches)
    for item in patch_dir:
        filenameMask, e = os.path.splitext(item)
        if(os.path.exists(check_dir + item)):
            image = cv2.imread(check_dir + filenameMask+".png",1)
            image_mask = cv2.imread(source_path+"test_mask\\" + filenameMask+".png",0)
            image_mask_1 = cv2.imread(source_path+"test_mask_1\\" + filenameMask+".png",0)
            cv2.imwrite(dst + "test/"+ filenameMask + ".png", image)
            cv2.imwrite(dst + "test_mask/" + filenameMask + ".png", image_mask)
            cv2.imwrite(dst+"test_mask_1/" + filenameMask + ".png", image_mask_1)

if __name__ == '__main__':
    # resize()
    # convert_four_channels_to_three()
    # mask_division()
    # crop()
    # labeling()
    # copy_mask_with_same_tumor_name_2()
    # create_empty_mask_for_patches()
    search_patch_in_folder()