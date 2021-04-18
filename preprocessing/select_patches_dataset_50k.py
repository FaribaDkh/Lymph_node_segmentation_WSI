import pandas as pd
import os,sys
import glob
import cv2
patch_all = "E:\\camelyon16\\50k_dataset\\"
# dst = "E:\\camelyon16\\50k_dataset\\tumor\\"
dst = "E:\\camelyon16\\50k_dataset\\normal_patches\\"
# dst_dir = os.listdir(dst)
# tumorDst_1 = "E:\\camelyon16\\level_2\\n_t_mask_1\\"
allPatchesdirs = os.listdir(patch_all+"/tumor/")
for item in allPatchesdirs:
    file_name, e = os.path.splitext(item)
    parts = file_name.split('_')
    img = cv2.imread(patch_all+"/tumor/"+item,1)
    # if os.path.isfile([patch_all+"/normal_mask/" + item]):
    im_mask = cv2.imread(patch_all+"/tumor_mask/" + file_name + "_mask.png", 1)
    im_mask_1 = cv2.imread(patch_all+"/tumor_mask_1/" + file_name + "_mask_1.png", 1)
    cv2.imwrite(dst+ "\\tumor_mask\\"+file_name+".png",im_mask)
    cv2.imwrite(dst+ "\\tumor_mask_1\\"+file_name+".png",im_mask_1)
    cv2.imwrite(dst+ "\\tumor\\"+file_name+".png",img)