import pandas as pd
import os,sys
import glob
patch_all = "E:\\camelyon16\\all_patches\\"
# dst = "E:\\camelyon16\\50k_dataset\\tumor\\"
dst = "E:\\camelyon16\\dataset_for_training\\level_1\\n_t\\"
dst_dir = os.listdir(dst)
# tumorDst_1 = "E:\\camelyon16\\level_2\\n_t_mask_1\\"
allPatchesdirs = os.listdir(patch_all)


def count_patch():
    df = pd.DataFrame({"slide_name": [], "count_extracted_patches": [], "count_random_selection_patches": []})
    # for tumor_slide in allPatchesdirs:
    #     filename, e = os.path.splitext(tumor_slide)
    #     file_name_list = filename.split('_')
    #     lens_of_list = len(file_name_list)
    #     if lens_of_list == 2:
    #         images = glob.glob(patch_all + "\\" + filename + "\\" + "*.png")
    #         len_all_patches = len(images)
            # df = df.append({"slide_name":filename,"count_extracted_patches":len_all_patches},ignore_index=True)
    for i in range(1,111):
        patch_count_slide_name = 0
        for item in dst_dir:
            filename, e = os.path.splitext(item)
            file_name_list = filename.split('_')
            slide_name = int(file_name_list[1])
            if slide_name == i:
                patch_count_slide_name += 1
        df = df.append({"slide_name": "tumor_"+str(i), "count_random_selection_patches": patch_count_slide_name}, ignore_index=True)
    df.to_csv("df_count_patches.csv")
if __name__ == '__main__':
    count_patch()

