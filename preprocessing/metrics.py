import pandas as pd
# Import matplotlib to setup the figure size of box plot
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set(style="whitegrid")

# Load the dataset from a CSV file
# df = pd.read_csv("E:\\camelyon16\\dataset_for_training\\10k_dataset\\10k_level0_balanced\\" + "bn_vs_wobn.csv")
source_path = "E:\\camelyon16\\50k_dataset\\dataset_splitted\\test_separate\\"
# df_1 = pd.read_csv(source_path+"dataset_splitted\\" + "50k_6_exp.csv")
df_1 = pd.read_csv(source_path+ "normal.csv")
# df_2 = pd.read_csv("E:\\camelyon16\\dataset_for_training\\10k_dataset\\10k_level0_balanced\\" + "with_bn.csv")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def make_labels(ax, boxplot):

    # Grab the relevant Line2D instances from the boxplot dictionary
    iqr = boxplot['boxes'][0]
    caps = boxplot['caps']
    med = boxplot['medians'][0]
    fly = boxplot['fliers'][0]

    # The x position of the median line
    xpos = med.get_xdata()

    # Lets make the text have a horizontal offset which is some
    # fraction of the width of the box
    xoff = 0.10 * (xpos[1] - xpos[0])

    # The x position of the labels
    xlabel = xpos[1] + xoff

    # The median is the y-position of the median line
    median = med.get_ydata()[1]

    # The 25th and 75th percentiles are found from the
    # top and bottom (max and min) of the box
    pc25 = iqr.get_ydata().min()
    pc75 = iqr.get_ydata().max()

    # The caps give the vertical position of the ends of the whiskers
    capbottom = caps[0].get_ydata()[0]
    captop = caps[1].get_ydata()[0]

    # Make some labels on the figure using the values derived above
    ax.text(xlabel, median,
            'Median = {:6.3g}'.format(median), va='center')
    ax.text(xlabel, pc25,
            '25th percentile = {:6.3g}'.format(pc25), va='center')
    ax.text(xlabel, pc75,
            '75th percentile = {:6.3g}'.format(pc75), va='center')
    ax.text(xlabel, capbottom,
            'Bottom cap = {:6.3g}'.format(capbottom), va='center')
    ax.text(xlabel, captop,
            'Top cap = {:6.3g}'.format(captop), va='center')

    # Many fliers, so we loop over them and create a label for each one
    for flier in fly.get_ydata():
        ax.text(1 + xoff, flier,
                'Flier = {:6.3g}'.format(flier), va='center')

df_metrics = pd.DataFrame({"slide_name": [], "TP": [],"TN":[],"FP":[],"FN":[],"Accuracy":[],"Sensitivity":[],
                           "specificity":[]," false positive rate":[], "dice_coeff": [], "jacard_index": [], "F1":[]})

import matplotlib.pyplot as plt

def Quratile_calculation(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    whis_1 = Q1 - 0.5 * IQR
    whis_2 = Q3 + 0.5 * IQR
    return (whis_1,whis_2)


def make_labels(ax, boxplot):

    # Grab the relevant Line2D instances from the boxplot dictionary
    iqr = boxplot['boxes'][0]
    caps = boxplot['caps']
    med = boxplot['medians'][0]
    fly = boxplot['fliers'][0]

    # The x position of the median line
    xpos = med.get_xdata()

    # Lets make the text have a horizontal offset which is some
    # fraction of the width of the box
    xoff = 0.10 * (xpos[1] - xpos[0])

    # The x position of the labels
    xlabel = xpos[1] + xoff

    # The median is the y-position of the median line
    median = med.get_ydata()[1]

    # The 25th and 75th percentiles are found from the
    # top and bottom (max and min) of the box
    pc25 = iqr.get_ydata().min()
    pc75 = iqr.get_ydata().max()

    # The caps give the vertical position of the ends of the whiskers
    capbottom = caps[0].get_ydata()[0]
    captop = caps[1].get_ydata()[0]
    print("median:",median)
    print("capbottom:",capbottom)
    print("captop:",captop)
    print("min of box:",pc25)
    print("max of box:",pc75)
    # Make some labels on the figure using the values derived above

# Make the figure

name = df_1['File_name']
bn = df_1['Type']
FP = df_1['FP'].astype('float')
FN = df_1['FN'].astype('float')
TP = df_1['TP'].astype('float')
TN = df_1['TN'].astype('float')

TP_rate = TP/(TP+FN)
FN_rate = FN/(TP+FN)
TN_rate = TN/(TN+FP)
FP_rate = FP/(TN+FP)
ACC = (TP+TN)/(TP+TN+FP+FN)

df_TP_rate = pd.DataFrame({"name":name,"Type": bn, "TN": TN,"TP": TP,"FP":FP,"FN": FN,"Metrics_value": TP_rate, "Metric_name":"TP_rate"})
df_FN_rate = pd.DataFrame({"name":name,"Type": bn, "TN": TN,"TP": TP,"FP":FP,"FN": FN,"Metrics_value": FN_rate, "Metric_name":"FN_rate"})
df_TN_rate= pd.DataFrame({"name":name,"Type": bn, "TN": TN,"TP": TP,"FP":FP,"FN": FN,"Metrics_value": TN_rate, "Metric_name":"TN_rate"})
df_FP_rate = pd.DataFrame({"name":name,"Type": bn, "TN": TN,"TP": TP,"FP":FP,"FN": FN,"Metrics_value": FP_rate, "Metric_name":"FP_rate"})
df_ACC = pd.DataFrame({"name":name,"Type": bn, "TN": TN,"TP": TP,"FP":FP,"FN": FN,"Metrics_value": ACC, "Metric_name":"ACCURACY"})
df_all = pd.DataFrame({"name":[],"whis1":[],"whis2":[],"outliers_count":[]})

### with_BN
# df_FP_w_B = df_FP_rate[(df_FP_rate['Metrics_value'] >= 0) & (df_FP_rate['BN'] == "50k_with_BN")]
# whis1_FP_w_B,whis2_FP_w_B = Quratile_calculation(df_FP_w_B['Metrics_value'])
# print("whis1_FP: ",whis1_FP_w_B)
# print("whis2_FP: ",whis2_FP_w_B)
# df_fp_w_B_count = df_FP_w_B.loc[(df_FP_w_B['Metrics_value'] > whis2_FP_w_B)].count()
# df_all = df_all.append({"name":"df_fp_w_B_count","whis1":whis1_FP_w_B,"whis2":whis2_FP_w_B,"outliers_count":df_fp_w_B_count[0]},ignore_index=True)
# df_fp_w_B = df_FP_w_B.loc[(df_FP_w_B['Metrics_value'] > whis2_FP_w_B)].to_csv('df_fp_w_B.csv')
#
# df_TP_w_B = df_TP_rate[(df_TP_rate['Metrics_value'] >= 0) & (df_TP_rate['BN'] == "50k_with_BN")]
# whis1_TP_w_B,whis2_TP_w_B = Quratile_calculation(df_TP_w_B['Metrics_value'])
# print("whis1_FP: ",whis1_TP_w_B)
# print("whis2_FP: ",whis2_TP_w_B)
# df_TP_w_B_count = df_TP_w_B.loc[(df_TP_w_B['Metrics_value'] > whis2_TP_w_B)|(df_TP_w_B['Metrics_value'] < whis1_TP_w_B)].count()
# df_high_TP_w_B = df_TP_w_B.loc[(df_TP_w_B['Metrics_value'] > 0.90)].to_csv('df_high_TP_w_B.csv')
# df_all = df_all.append({"name":"df_TP_w_B_count","whis1":whis1_TP_w_B,"whis2":whis2_TP_w_B,"outliers_count":df_TP_w_B_count[0]},ignore_index=True)
# df_TP_w_B.loc[(df_TP_w_B['Metrics_value'] > whis2_TP_w_B)|(df_TP_w_B['Metrics_value'] < whis1_TP_w_B)].to_csv('df_TP_w_B.csv')
#
#
# df_TN_w_B = df_TN_rate[(df_TN_rate['Metrics_value'] >= 0) & (df_TN_rate['BN'] == "50k_with_BN")]
# whis1_TN_w_B,whis2_TN_w_B = Quratile_calculation(df_TN_w_B['Metrics_value'])
# print("whis1_FP: ",whis1_TN_w_B)
# print("whis2_FP: ",whis2_TN_w_B)
# df_TN_w_B_count = df_TN_w_B.loc[(df_TN_w_B['Metrics_value'] > whis2_TN_w_B)|(df_TN_w_B['Metrics_value'] < whis1_TN_w_B)].count()
# df_all = df_all.append({"name":"df_TN_w_B","whis1": whis1_TN_w_B,"whis2":whis2_TN_w_B, "outliers_count":df_TN_w_B_count[0]},ignore_index=True)
# df_TN_w_B.loc[(df_TN_w_B['Metrics_value'] > whis2_TN_w_B)|(df_TN_w_B['Metrics_value'] < whis1_TN_w_B)].to_csv('df_TN_w_B.csv')
# df_high_TN_w_B = df_TN_w_B.loc[(df_TN_w_B['Metrics_value'] ==1)].to_csv('df_high_TN_w_B.csv')
#
# df_FN_w_B = df_FN_rate[(df_FN_rate['Metrics_value'] >= 0) & (df_FN_rate['BN'] == "50k_with_BN")]
# whis1_FN_w_B,whis2_FN_w_B = Quratile_calculation(df_FN_w_B['Metrics_value'])
# print("whis1_FP: ",whis1_FN_w_B)
# print("whis2_FP: ",whis2_FN_w_B)
# df_FN_w_B_count = df_FN_w_B.loc[(df_FN_w_B['Metrics_value'] > whis2_FN_w_B)|(df_FN_w_B['Metrics_value'] < whis1_FN_w_B)].count()
# df_all = df_all.append({"name":"df_FN_w_B_count","whis1":whis1_FN_w_B,"whis2":whis2_FN_w_B,"outliers_count":df_FN_w_B_count[0]},ignore_index=True)
# df_FN_w_B = df_FN_w_B.loc[(df_FN_w_B['Metrics_value'] > whis2_FN_w_B)|(df_FN_w_B['Metrics_value'] < whis1_FN_w_B)].to_csv('df_FN_w_B.csv')
#
# ### without_BN
# df_FP_w_O_B = df_FP_rate[(df_FP_rate['Metrics_value'] >= 0) & (df_FP_rate['BN'] == "50k_without_BN")]
# whis1_FP_w_O_B,whis2_FP_w_O_B = Quratile_calculation(df_FP_w_O_B['Metrics_value'])
# print("whis1_FP: ",whis1_FP_w_O_B)
# print("whis2_FP: ",whis2_FP_w_O_B)
# df_fp_with_out_BN_count = df_FP_w_O_B.loc[(df_FP_w_O_B['Metrics_value'] > whis2_FP_w_O_B)].count()
# df_all = df_all.append({"name":"df_fp_with_out_BN_count","whis1":whis1_FP_w_O_B,"whis2":whis2_FP_w_O_B,"outliers_count":df_fp_with_out_BN_count[0]},ignore_index=True)
# df_fp_with_out_BN = df_FP_w_O_B.loc[(df_FP_w_O_B['Metrics_value'] > whis2_FP_w_O_B)].to_csv('df_fp_with_out_BN.csv')
#
# df_TP_w_O_B = df_TP_rate[(df_TP_rate['Metrics_value'] >= 0) & (df_TP_rate['BN'] == "50k_without_BN")]
# whis1_TP_w_O_B,whis2_TP_w_O_B = Quratile_calculation(df_TP_w_O_B['Metrics_value'])
# print("whis1_FP: ",whis1_TP_w_O_B)
# print("whis2_FP: ",whis2_TP_w_O_B)
# df_TP_with_out_BN_count = df_TP_w_O_B.loc[(df_TP_w_O_B['Metrics_value'] > whis2_TP_w_O_B)|(df_TP_w_O_B['Metrics_value'] < whis1_TP_w_O_B)].count()
# df_all = df_all.append({"name":"df_TP_with_out_BN_count","whis1":whis1_TP_w_O_B,"whis2":whis2_TP_w_O_B,"outliers_count":df_TP_with_out_BN_count[0]},ignore_index=True)
# df_TP_with_out_BN = df_TP_w_O_B.loc[(df_TP_w_O_B['Metrics_value'] > whis2_TP_w_O_B)|(df_TP_w_O_B['Metrics_value'] < whis1_TP_w_O_B)].to_csv('df_TP_with_out_BN.csv')
#
# df_TN_w_O_B = df_TN_rate[(df_TN_rate['Metrics_value'] >= 0) & (df_TN_rate['BN'] == "50k_without_BN")]
# whis1_TN_w_O_B,whis2_TN_w_O_B = Quratile_calculation(df_TN_w_O_B['Metrics_value'])
# print("whis1_FP: ",whis1_TN_w_O_B)
# print("whis2_FP: ",whis2_TN_w_O_B)
# df_TN_with_out_BN_count = df_TN_w_O_B.loc[(df_TN_w_O_B['Metrics_value'] > whis2_TN_w_O_B)|(df_TN_w_O_B['Metrics_value'] < whis1_TN_w_O_B)].count()
# df_all = df_all.append({"name":"df_TN_with_out_BN_count","whis1":whis1_TN_w_O_B,"whis2":whis2_TN_w_O_B,"outliers_count":df_TN_with_out_BN_count[0]},ignore_index=True)
# df_TN_with_out_BN = df_TN_w_O_B.loc[(df_TN_w_O_B['Metrics_value'] > whis2_TN_w_O_B)|(df_TN_w_O_B['Metrics_value'] < whis1_TN_w_O_B)].to_csv('df_TN_with_out_BN.csv')
#
# df_FN_w_O_B = df_FN_rate[(df_FN_rate['Metrics_value'] >= 0) & (df_FN_rate['BN'] == "50k_without_BN")]
# whis1_FN_w_O_B,whis2_FN_w_O_B = Quratile_calculation(df_FN_w_O_B['Metrics_value'])
# print("whis1_FP: ",whis1_FN_w_O_B)
# print("whis2_FP: ",whis2_FN_w_O_B)
# df_FN_with_out_BN_count = df_FN_w_O_B.loc[(df_FN_w_O_B['Metrics_value'] > whis2_FN_w_O_B)|(df_FN_w_O_B['Metrics_value'] < whis1_FN_w_O_B) ].count()
# df_all = df_all.append({"name":"df_FN_with_out_BN_count","whis1":whis1_FN_w_O_B,"whis2":whis2_FN_w_O_B,"outliers_count":df_FN_with_out_BN_count[0]},ignore_index=True)
# df_FN_with_out_BN = df_FN_w_O_B.loc[(df_FN_w_O_B['Metrics_value'] > whis2_FN_w_O_B)|(df_FN_w_O_B['Metrics_value'] < whis1_FN_w_O_B) ].to_csv('df_FN_with_out_BN.csv')
# df_all.to_csv("df_all.csv")
#
# df_FP_rate.loc[(df_FP_rate['Metrics_value'] > 0.5) & (df_FP_rate['Type'] == "50k_TransUnet")].to_csv('high_FP_50k_TransUnet.csv')
# df_FN_rate.loc[(df_FN_rate['Metrics_value'] > 0.5) & (df_FP_rate['Type'] == "50k_TransUnet")].to_csv('high_FN_50k_TransUnet.csv')
# df_FP_rate.loc[(df_FP_rate['Metrics_value'] > 0.5) & (df_FP_rate['BN'] == "50k_without_BN")].to_csv('df_FP_with_out_bn.csv')
# df_TP_rate.loc[(df_TP_rate['Metrics_value'] < 0.5) & (df_TP_rate['BN'] == "50k_with_BN")].to_csv('df_TP_with_bn.csv')
# df_TP_rate.loc[(df_TP_rate['Metrics_value'] < 0.5) & (df_TP_rate['BN'] == "50k_without_BN")].to_csv('df_TP_with_out_bn.csv')
# df_FP_rate.loc[(df_FP_rate['Metrics_value'] > 0.7) & (df_FP_rate['BN'] == "50k_without_BN")].to_csv('FPeq1wbn.csv')
# df_FP_rate.loc[(df_FP_rate['Metrics_value'] > 0.7) & (df_FP_rate['BN'] == "50k_with_BN")].to_csv('fpeqwobn.csv')
# df_ACC.loc[(df_ACC['Metrics_value'] < 0.6) & (df_FP_rate['BN'] == "50k_with_BN")].to_csv('df_ACC_less_than_60.csv')
# df_FP_rate.to_csv('df_FP_rate.csv')
# df_FN_rate = df_FN_rate.loc[(df_FN_rate['Metrics_value'] !=1)]
# df_FP_rate = df_FP_rate.loc[(df_FP_rate['Metrics_value'] !=1)]
# df_TP_rate = df_TP_rate.loc[(df_TP_rate['Metrics_value'] !=0)]
# df_TN_rate = df_TN_rate.loc[(df_TN_rate['Metrics_value'] !=0)]
# Outliers
path = "E:\\camelyon16\\new_dataset\\splitted_dataset\\"
value = 0.75
# df_TP_rate.loc[(df_TP_rate['Metrics_value'] < value) & (df_TP_rate['Type'] == "Three_layers_w_BN")].to_csv(path+'low_df_ACC_50k_TransUnet.csv')
# df_TP_rate.loc[(df_TP_rate['Metrics_value'] <value) & (df_TP_rate['Type'] == "50k_with_BN")].to_csv(path+'low_df_ACC_BN.csv')
# df_TP_rate.loc[(df_TP_rate['Metrics_value'] < value) & (df_TP_rate['Type'] == "Reinhard_center1")].to_csv(path+'low_df_ACC_50k_reinhard_center1.csv')
# df_TP_rate.loc[(df_TP_rate['Metrics_value'] < value) & (df_TP_rate['Type'] == "Reinhard_center2")].to_csv(path+'low_df_ACC_50k_reinhard_center2.csv')
# df_TP_rate.loc[(df_TP_rate['Metrics_value'] < value) & (df_TP_rate['Type'] == "Reinhard_center2")].to_csv(path+'low_df_ACC_50k_reinhard_center2.csv')
# df_TP_rate.loc[(df_TP_rate['Metrics_value'] < value) & (df_TP_rate['Type'] == "Reinhard_center2")].to_csv(path+'low_df_ACC_50k_reinhard_center2.csv')
#
# Five_layers_with_BN = df_TP_rate.loc[(df_TP_rate['Metrics_value'] < value) & (df_TP_rate['Type'] == "Five_layers_with_BN")].count()
# Five_layers_without_BN = df_TP_rate.loc[(df_TP_rate['Metrics_value'] < value) & (df_TP_rate['Type'] == "Five_layers_without_BN")].count()
# low_df_ACC_50k_reinhard_center1 = df_TP_rate.loc[(df_TP_rate['Metrics_value'] <value) & (df_TP_rate['Type'] == "Five_layers_with_BN_RH_center1")].count()
# low_df_ACC_50k_reinhard_center2 = df_TP_rate.loc[(df_TP_rate['Metrics_value'] < value) & (df_TP_rate['Type'] == "Five_layers_with_BN_RH_center2")].count()
# Three_layers_w_BN = df_TP_rate.loc[(df_TP_rate['Metrics_value'] < value) & (df_TP_rate['Type'] == "Three_layers_w_BN")].count()
# Three_layers_w_BN = df_TP_rate.loc[(df_TP_rate['Metrics_value'] < value) & (df_TP_rate['Type'] == "Three_layers_w_BN")].count()
# Three_layers_w_BN = df_TP_rate.loc[(df_TP_rate['Metrics_value'] < value) & (df_TP_rate['Type'] == "Three_layers_w_BN")].count()


# print("Five_layers_with_BN: ",Five_layers_with_BN[0])
# print("Five_layers_without_BN: ",Five_layers_without_BN[0])
# print("low_df_ACC_50k_reinhard_center1: ",low_df_ACC_50k_reinhard_center1[0])
# print("low_df_ACC_50k_reinhard_center2: ",low_df_ACC_50k_reinhard_center2[0])
# print("Three_layers_w_BN: ",Three_layers_w_BN[0])
cdf_new = pd.concat([df_TN_rate,df_FP_rate,df_ACC], axis=0)
# accuracy for normal patches
df_ACC.loc[(df_ACC['Metrics_value'] < value) & (df_ACC['Type'] == "Three_layers_w_BN")].to_csv(path+'low_df_ACC_50k_TransUnet.csv')
df_ACC.loc[(df_ACC['Metrics_value'] <value) & (df_ACC['Type'] == "50k_with_BN")].to_csv(path+'low_df_ACC_BN.csv')
df_ACC.loc[(df_ACC['Metrics_value'] < value) & (df_ACC['Type'] == "Reinhard_center1")].to_csv(path+'low_df_ACC_50k_reinhard_center1.csv')
df_ACC.loc[(df_ACC['Metrics_value'] < value) & (df_ACC['Type'] == "Reinhard_center2")].to_csv(path+'low_df_ACC_50k_reinhard_center2.csv')
df_ACC.loc[(df_ACC['Metrics_value'] < value) & (df_ACC['Type'] == "Reinhard_center2")].to_csv(path+'low_df_ACC_50k_reinhard_center2.csv')
df_ACC.loc[(df_ACC['Metrics_value'] < value) & (df_ACC['Type'] == "Reinhard_center2")].to_csv(path+'low_df_ACC_50k_reinhard_center2.csv')

Five_layers_with_BN = df_ACC.loc[(df_ACC['Metrics_value'] < value) & (df_ACC['Type'] == "Five_layers_with_BN")].count()
Five_layers_without_BN = df_ACC.loc[(df_ACC['Metrics_value'] < value) & (df_ACC['Type'] == "Five_layers_without_BN")].count()
low_df_ACC_50k_reinhard_center1 = df_ACC.loc[(df_ACC['Metrics_value'] <value) & (df_ACC['Type'] == "Five_layers_with_BN_RH_center1")].count()
low_df_ACC_50k_reinhard_center2 = df_ACC.loc[(df_ACC['Metrics_value'] < value) & (df_ACC['Type'] == "Five_layers_with_BN_RH_center2")].count()
Three_layers_w_BN = df_ACC.loc[(df_ACC['Metrics_value'] < value) & (df_ACC['Type'] == "Three_layers_w_BN")].count()
Three_layers_w_BN = df_ACC.loc[(df_ACC['Metrics_value'] < value) & (df_ACC['Type'] == "Three_layers_w_BN")].count()
Three_layers_w_BN = df_ACC.loc[(df_ACC['Metrics_value'] < value) & (df_ACC['Type'] == "Three_layers_w_BN")].count()
# cdf_new = pd.concat([df_TP_rate,df_FN_rate,df_ACC], axis=0)
# cdf_new = pd.concat([df_TN_rate,df_FN_rate,df_TP_rate,df_FP_rate,df_ACC], axis=0)
# cdf_new.to_csv('df_metrics_normal.csv')
# cdf_new = cdf_new.dropna()
g = sns.catplot(hue="Type", y="Metrics_value",
                x="Metric_name",
                data=cdf_new, kind="box");

plt.show()
# g = sns.catplot(hue="BN", y="Metrics_value",
#                 x="Metric_name",
#                 data=cdf_sns, kind= "box",palette="Oranges_r");
#
# plt.show()

