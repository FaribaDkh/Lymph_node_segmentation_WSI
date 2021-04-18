import cv2 as cv
import os
import sys
import glob
from PIL import Image
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
import cv2

slide_patch_name = "E:\\camelyon16\\50k_dataset\\dataset_splitted\\test\\"
# source_path = "E:\\camelyon16\\dataset_for_training\\level_2\\dataset\\all_tumor\\test\\"
# source_path = "E:\\camelyon16\\dataset_for_training\\10k_dataset\\test_cam2017\\"
source_path = "E:\\camelyon16\\50k_dataset\\dataset_splitted\\"
patch_dir = "E:\\camelyon16\\50k_dataset\\dataset_splitted\\test\\"
# GT_path = "E:\\camelyon16\\dataset_for_training\\all_tumor_patches_test\\masks\\"
GT_path = source_path + "\\test_mask\\"
results_path = source_path +"new\\"
allresults_path_dir = os.listdir(results_path)
allPatchesdirs = os.listdir(patch_dir)
all_slide_patch_dirs = os.listdir(slide_patch_name)
maskdstDir = os.listdir(GT_path)
images = glob.glob(patch_dir + "*.png")
def combine_3():

    for item in allPatchesdirs:
        fileNameTumor, e = os.path.splitext(item)
        print(item)
        img = cv.imread(source_path + "\\test\\" + fileNameTumor+".png", 1)
        # img_cycleGan_a = cv.imread(source_path + "cycleGan\\test_a\\" + fileNameTumor + ".png", 1)
        # img_cycleGan_b = cv.imread(source_path + "cycleGan\\test_b\\" + fileNameTumor + ".png", 1)
        mask = cv.imread(source_path + "\\test_mask\\" + fileNameTumor+".png", 1)
        TransUnet = cv.imread(source_path + "\\new\\" + fileNameTumor + ".png", 1)
        # org = cv.imread(source_path + "dataset\\results\\visualize\\" + fileNameTumor+".png", 1)
        Unet = cv.imread(source_path + "\\output_50k_wbn\\visualize\\" + fileNameTumor + ".png", 1)
        # cycleGan_b = cv.imread(source_path + "cycleGan\\results_b\\visualize\\" + fileNameTumor + ".png", 1)
        # reinhard_mask = cv.imread(source_path + "Reinhard\\results\\visualize\\" + fileNameTumor + ".png", 1)
        mask = mask * 255
        TransUnet = TransUnet * 255

        img.astype(np.int32)
        # org.astype(np.int32)
        # cycleGan_a.astype(np.int32)
        # cycleGan_b.astype(np.int32)

        fig, ax = plt.subplots(1, 4, figsize=(15, 9))

        ax[0].imshow(img)
        cmap = plt.cm.jet
        cmap.set_bad('w', 1.)
        ax[1].imshow(mask)
        ax[2].imshow(Unet)
        TransUnet.astype(np.int32)
        img.astype(np.int32)
        dst = cv2.addWeighted(TransUnet, 0.5, img, 0.5, 0.0, dtype=cv2.CV_32F)
        dst = dst.astype(np.uint8)
        ax[3].imshow(dst)
        # ax[4].imshow(cycleGan_a)
        # ax[5].imshow(img_cycleGan_b)
        # ax[6].imshow(cycleGan_b)
        # ax[7].imshow(reinhard)
        # ax[8].imshow(reinhard_mask)

        ax[0].title.set_text('image')
        ax[1].title.set_text('mask')
        ax[2].title.set_text( 'unet')
        ax[3].title.set_text('TransUnet')
        # ax[4].title.set_text('Result C_a')
        # ax[5].title.set_text('CycleGAN C_b')
        # ax[6].title.set_text('Result C_b')
        # ax[7].title.set_text('Reinhard')
        # ax[8].title.set_text('Result Reinhard')

        plt.savefig(
            source_path + "\\combine\\"\
            + fileNameTumor + ".png", bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        plt.clf()
        plt.cla()
def combine_2():

    for item in allPatchesdirs:
        fileNameTumor, e = os.path.splitext(item)
        print(item)
        level0 = cv.imread(source_path+"level0\\imgs\\" + fileNameTumor+".png", 1)
        level1 = cv.imread(source_path+"level1\\imgs\\" + fileNameTumor+".png", 1)
        level2_256 = cv.imread(source_path+"level2_256\\imgs\\" + fileNameTumor+".png", 1)
        level2 = cv.imread(source_path+"level2\\imgs\\" + fileNameTumor+".png", 1)
        level0_mask = cv.imread(source_path+"level0\\masks_1\\"+ fileNameTumor+".png",1)
        level0_mask = level0_mask * 255
        level1_mask = cv.imread(source_path+"level1\\masks_1\\" + fileNameTumor+".png", 1)
        level1_mask = level1_mask * 255
        level2_256_mask = cv.imread(source_path+"level2_256\\masks_1\\" + fileNameTumor+".png", 1)
        level2_256_mask = level2_256_mask * 255
        level2_mask = cv.imread(source_path+"level2\\masks_1\\" + fileNameTumor+".png", 1)
        level2_mask = level2_mask * 255
        level0_result = cv.imread(source_path+"level0\\results\\visualize\\" + fileNameTumor+".png", 1)
        level1_result = cv.imread(source_path+"level1\\results\\visualize\\" + fileNameTumor+".png", 1)
        level2_256_result = cv.imread(source_path+"level2_256\\results\\visualize\\" + fileNameTumor+".png", 1)
        level2_result = cv.imread(source_path+"level2\\results\\visualize\\" + fileNameTumor+".png", 1)
        level0.astype(np.int32)
        level1.astype(np.int32)
        level2_256.astype(np.int32)
        level2.astype(np.int32)

        fig, ax = plt.subplots(3, 4, figsize=(10, 9))

        ax[0][0].imshow(level2)
        cmap = plt.cm.jet
        cmap.set_bad('w', 1.)
        ax[1][0].imshow(level2_mask)
        ax[2][0].imshow(level2_result)
        ax[0][0].title.set_text('10X 512(img_size)')
        ax[1][0].title.set_text('10X 512')
        ax[2][0].title.set_text('10X Results')

        ax[0][1].imshow(level2_256)
        cmap = plt.cm.jet
        cmap.set_bad('w', 1.)
        ax[1][1].imshow(level2_256_mask)
        ax[2][1].imshow(level2_256_result)
        ax[0][1].title.set_text('10X 256(img_size)')
        ax[1][1].title.set_text('10X GT')
        ax[2][1].title.set_text('10X Results')


        ax[0][2].imshow(level1)
        cmap = plt.cm.jet
        cmap.set_bad('w', 1.)
        ax[1][2].imshow(level1_mask)
        ax[2][2].imshow(level1_result)
        ax[0][2].title.set_text('20X')
        ax[1][2].title.set_text('20X GT')
        ax[2][2].title.set_text('20X Results')

        ax[0][3].imshow(level0)
        cmap = plt.cm.jet
        cmap.set_bad('w', 1.)
        ax[1][3].imshow(level0_mask)
        ax[2][3].imshow(level0_result)
        ax[0][3].title.set_text('40X')
        ax[1][3].title.set_text('40X GT')
        ax[2][3].title.set_text('40X Results')

        # ax[4].title.set_text('Overlaid output')
        plt.savefig(
            "E:\\camelyon16\\dataset_for_training\\10k_dataset\\test_cam2017\\all_results\\" \
            + fileNameTumor + ".png", bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        plt.clf()
        plt.cla()
def combine():
    for item in allPatchesdirs:
        fileNameTumor, e = os.path.splitext(item)
        original_image = cv.imread(source_path + fileNameTumor+".png", 1)
        im = cv.imread(results_path+"\\original\\visualize\\" + fileNameTumor + ".png", 1)
        gt = cv.imread(GT_path + fileNameTumor+".png", 1)
        im_RGBHist = cv.imread(results_path+"\\RGBHist\\visualize\\" + fileNameTumor + ".png", 1)
        im_enhanced_color = cv.imread(results_path+"\\enhanced_color\\visualize\\" + fileNameTumor+".png", 1)
        im_Rug = cv.imread(results_path+"\\Ruggedise\\visualize\\" + fileNameTumor+".png", 1)
        im_reinhard = cv.imread(results_path+"\\Reinhard\\visualize\\" + fileNameTumor+".png", 1)

        fig, ax = plt.subplots(2, 7, figsize=(10, 4))
        original_image.astype(np.int32)
        im.astype(np.int32)
        im_RGBHist.astype(np.int32)
        im_enhanced_color.astype(np.int32)
        im_Rug.astype(np.int32)
        im_reinhard.astype(np.int32)
        ax[0].imshow(original_image)
        ax[2].imshow(im)

        cmap = plt.cm.jet
        cmap.set_bad('w', 1.)
        ax[1].imshow(gt)
        cmap = plt.cm.jet
        cmap.set_bad('w', 1.)

        ax[3].imshow(im_RGBHist)
        ax[4].imshow(im_enhanced_color)
        ax[5].imshow(im_Rug)
        ax[6].imshow(im_reinhard)

        ax[0].title.set_text('Original Image')
        ax[2].title.set_text('Original result')
        ax[1].title.set_text('Ground Truth')
        ax[3].title.set_text('RGBHist')
        ax[4].title.set_text('enhanced_color')
        ax[5].title.set_text('Ruggedise')
        ax[6].title.set_text('Reinhard')
        # ax[4].title.set_text('Overlaid output')
        plt.savefig("E:\\camelyon16\\dataset_for_training\\all_tumor_patches_test\\result_color_normalization\\combine\\"\
                    + fileNameTumor + ".png", bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        plt.clf()
        plt.cla()
def select_patches():
    for item2 in all_slide_patch_dirs:
        path_name, e = os.path.splitext(item2)
        for item in allPatchesdirs:
            file_name, e =os.path.splitext(item)
            im = cv.imread(source_path+ path_name+"/"+file_name+".png",1)
            mask = cv.imread(source_path+ path_name+"/"+file_name+".png",0)
            # if
if __name__ == '__main__':
    combine_3()