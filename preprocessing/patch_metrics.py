import os, sys
import cv2
import numpy as np
import random
import glob
from PIL import Image
import os.path as osp
import pandas as pd

patch_all = "E:\\camelyon16\\50k_dataset\\dataset_splitted\\test\\"
dst = "E:\\camelyon16\\50k_dataset\\dataset_splitted\\test_separate\\"
mask_all = "E:\\camelyon16\\50k_dataset\\dataset_splitted\\test_mask_1\\"
# tumorDst_1 = "E:\\camelyon16\\level_2\\n_t_mask_1\\"
allPatchesdirs = os.listdir(mask_all)


def tumor_patches():
    for tumor_slide in allPatchesdirs:
        image_name, e = os.path.splitext(tumor_slide)
        image_mask = cv2.imread(mask_all + "\\" + image_name + ".png", 0) * 255
        imagem = ~image_mask
        image = cv2.imread(patch_all + "\\" + image_name + ".png", 1)
        # if (cv2.countNonZero(image_mask)>1):
        #     cv2.imwrite(dst + "\\" + "tumor" + "\\" + image_name + ".png", image)
        #     cv2.imwrite(dst + "\\" + "tumor_mask" + "\\" + image_name + ".png", image_mask)
        if(cv2.countNonZero(imagem)>1):
            cv2.imwrite(dst + "\\" + "normal" + "\\" + image_name + ".png", image)
            cv2.imwrite(dst + "\\" + "normal_mask" + "\\" + image_name + ".png", image_mask)


def csv_reader_patch():
    tumor_patches = "E:\\camelyon16\\50k_dataset\\dataset_splitted\\test_separate\\normal\\"
    tumor_patches_dir = os.listdir(tumor_patches)
    appended_data = []
    df = pd.read_csv("E:\\camelyon16\\50k_dataset\\dataset_splitted\\50k_6_exp.csv")

    for tumor_slide in tumor_patches_dir:
        image_name, e = os.path.splitext(tumor_slide)
        # new_file = df.loc[(df['File_name']==image_name)]
        appended_data.append(df.loc[(df['File_name']==image_name)])
    appended_data = pd.concat(appended_data)
    appended_data.to_csv(dst+ "normal_with_morethan_one_pixel.csv")
def metrics():
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    source_path = "E:\\camelyon16\\50k_dataset\\dataset_splitted\\test_separate\\"
    df_1 = pd.read_csv(source_path + "normal.csv")

    name = df_1['File_name']
    bn = df_1['Type']
    FP = df_1['FP'].astype('float')
    FN = df_1['FN'].astype('float')
    TP = df_1['TP'].astype('float')
    TN = df_1['TN'].astype('float')

    TP_rate = TP / (TP + FN)
    FN_rate = FN / (TP + FN)
    TN_rate = TN / (TN + FP)
    FP_rate = FP / (TN + FP)
    ACC = (TP + TN) / (TP + TN + FP + FN)

    df_TP_rate = pd.DataFrame(
        {"name": name, "Type": bn, "TN": TN, "TP": TP, "FP": FP, "FN": FN, "Metrics_value": TP_rate,
         "Metric_name": "TP_rate"})
    df_FN_rate = pd.DataFrame(
        {"name": name, "Type": bn, "TN": TN, "TP": TP, "FP": FP, "FN": FN, "Metrics_value": FN_rate,
         "Metric_name": "FN_rate"})
    df_TN_rate = pd.DataFrame(
        {"name": name, "Type": bn, "TN": TN, "TP": TP, "FP": FP, "FN": FN, "Metrics_value": TN_rate,
         "Metric_name": "TN_rate"})
    df_FP_rate = pd.DataFrame(
        {"name": name, "Type": bn, "TN": TN, "TP": TP, "FP": FP, "FN": FN, "Metrics_value": FP_rate,
         "Metric_name": "FP_rate"})
    df_ACC = pd.DataFrame({"name": name, "Type": bn, "TN": TN, "TP": TP, "FP": FP, "FN": FN, "Metrics_value": ACC,
                           "Metric_name": "ACCURACY"})

    path = "E:\\camelyon16\\new_dataset\\splitted_dataset\\"
    value = 0.75

    cdf_new = pd.concat([df_TN_rate, df_FP_rate, df_ACC], axis=0)
    # accuracy for normal patches
    cdf_new.loc[(cdf_new['Metrics_value'] < value) & (cdf_new['Type'] == "Three_layers_w_BN") & (cdf_new['Metric_name'] == "ACCURACY")]\
        .to_csv(path + 'low_df_ACC_50k_TransUnet.csv')
    cdf_new.loc[(cdf_new['Metrics_value'] < value) & (cdf_new['Type'] == "50k_with_BN")& (cdf_new['Metric_name'] == "ACCURACY")]\
        .to_csv(path + 'low_df_ACC_BN.csv')
    cdf_new.loc[(cdf_new['Metrics_value'] < value) & (cdf_new['Type'] == "Reinhard_center1")]\
        .to_csv(path + 'low_df_ACC_50k_reinhard_center1.csv')
    cdf_new.loc[(cdf_new['Metrics_value'] < value) & (cdf_new['Type'] == "Reinhard_center2")& (cdf_new['Metric_name'] == "ACCURACY")].to_csv(
        path + 'low_df_ACC_50k_reinhard_center2.csv')
    cdf_new.loc[(cdf_new['Metrics_value'] < value) & (cdf_new['Type'] == "50k_TransUnet")& (cdf_new['Metric_name'] == "ACCURACY")].to_csv(
        path + '50k_TransUnet.csv')
    cdf_new.loc[(cdf_new['Metrics_value'] < value) & (cdf_new['Type'] == "Kmeans_Five_layers_with_BN")& (cdf_new['Metric_name'] == "ACCURACY")].to_csv(
        path + 'Kmeans_Five_layers_with_BN.csv')

    Five_layers_with_BN = cdf_new.loc[
        (cdf_new['Metrics_value'] < value) & (cdf_new['Type'] == "Five_layers_with_BN")& (cdf_new['Metric_name'] == "ACCURACY")].count()
    Five_layers_without_BN = cdf_new.loc[
        (cdf_new['Metrics_value'] < value) & (cdf_new['Type'] == "Five_layers_without_BN")& (cdf_new['Metric_name'] == "ACCURACY")].count()
    low_df_ACC_50k_reinhard_center1 = cdf_new.loc[
        (cdf_new['Metrics_value'] < value) & (cdf_new['Type'] == "Five_layers_with_BN_RH_center1")& (cdf_new['Metric_name'] == "ACCURACY")].count()
    low_df_ACC_50k_reinhard_center2 = cdf_new.loc[
        (cdf_new['Metrics_value'] < value) & (cdf_new['Type'] == "Five_layers_with_BN_RH_center2")& (cdf_new['Metric_name'] == "ACCURACY")].count()
    Three_layers_w_BN = cdf_new.loc[(cdf_new['Metrics_value'] < value) & (cdf_new['Type'] == "Three_layers_w_BN")& (cdf_new['Metric_name'] == "ACCURACY")].count()
    TransUnet = cdf_new.loc[(cdf_new['Metrics_value'] < value) & (cdf_new['Type'] == "50k_TransUnet")& (cdf_new['Metric_name'] == "ACCURACY")].count()
    Kmeans_Five_layers_with_BN = cdf_new.loc[(cdf_new['Metrics_value'] < value) & (cdf_new['Type'] == "Kmeans_Five_layers_with_BN")& (cdf_new['Metric_name'] == "ACCURACY")].count()
    print("Five_layers_with_BN: ",Five_layers_with_BN[0])
    print("Five_layers_without_BN: ",Five_layers_without_BN[0])
    print("low_df_ACC_50k_reinhard_center1: ",low_df_ACC_50k_reinhard_center1[0])
    print("low_df_ACC_50k_reinhard_center2: ",low_df_ACC_50k_reinhard_center2[0])
    print("Three_layers_w_BN: ",Three_layers_w_BN[0])
    print("TransUnet: ",TransUnet[0])
    print("Kmeans_Five_layers_with_BN: ",Kmeans_Five_layers_with_BN[0])
    # cdf_new = pd.concat([df_TP_rate,df_FN_rate,df_ACC], axis=0)
    # cdf_new = pd.concat([df_TN_rate,df_FN_rate,df_TP_rate,df_FP_rate,df_ACC], axis=0)
    # cdf_new.to_csv('df_metrics_normal.csv')
    # cdf_new = cdf_new.dropna()
    g = sns.catplot(hue="Type", y="Metrics_value",
                    x="Metric_name",
                    data=cdf_new, kind="box");

    plt.show()


if __name__ == '__main__':
    # tumor_patches()
    # csv_reader_patch()
    metrics()