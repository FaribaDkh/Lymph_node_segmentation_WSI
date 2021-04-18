import pandas as pd
# Import matplotlib to setup the figure size of box plot
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set(style="whitegrid")

# Load the dataset from a CSV file
# df = pd.read_csv("E:\\camelyon16\\dataset_for_training\\10k_dataset\\10k_level0_balanced\\" + "bn_vs_wobn.csv")
df_1 = pd.read_csv("E:\\camelyon16\\dataset_for_training\\10k_dataset\\10k_level0_balanced\\" + "10k_unet_with_bn_each_patch.csv")
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

# g = sns.catplot(x="Bn",y = "percentage",
#                 hue="metrics",
#                 data=df, kind="box",
#                 height=4, aspect=.7)

# plt.show()
# sns.boxplot(y='jacard_index',x = "image_type" ,hue='level',data=df,palette="Set3",whis =1.6,width =0.6)
# plt.show()
# sns.boxplot(y='F1',x = "image_type" ,hue='level',data=df,palette="Set3",whis =1.6,width =0.6)
# plt.show()
import numpy as np
# df.index = np.arange(1, len(df) + 1)

# df = df.replace(0, np.NaN)
# df.mean()
import matplotlib.pyplot as plt
import numpy as np


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

name = df_1['filename']
bn = df_1['Bn']
FP = df_1['FP'].astype('float')
FN = df_1['FN'].astype('float')
TP = df_1['TP'].astype('float')
TN = df_1['TN'].astype('float')

# sensitivity, recall, hit rate, or true positive rate (TPR)
sensitivity = TP /(TP + FN)
# specificity, selectivity or true negative rate (TNR)
specificity = TN/(TN+FP)
# precision or positive predictive value (PPV)
precision = TP/(TP+FP)
# negative predictive value (NPV)
NPV = TN/(TN+FN)
# false discovery rate (FDR)
FDR = FP/(FP+TP)
# miss rate or false negative rate (FNR)
FNR = FN/(FN+TP)
# fall-out or false positive rate (FPR)
FPR = FP/(FP+TN)
# false omission rate (FOR)
FOR = FN/(FN+TN)
# accuracy (ACC)
ACC = (TP+TN)/(TP+TN+FP+FN)
dice_coef = 2*TP/((TP+FP)+(TP+FN))



# Matthews correlation coefficient (MCC)
MCC = ((TP*TN)-(FP*FN))/((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)**(1/2))
# Fowlkes–Mallows index (FM)
FM = (precision*sensitivity)**(1/2)

mrg_a = ((TP + FN) * (TP + FP)) / (TP + FN + FP + TN)
mrg_b = ((FP + TN) * (FN + TN)) / (TP + FN + FP + TN)
expec_agree = (mrg_a + mrg_b) / (TP + FN + FP + TN)
obs_agree = (TP +TN) / (TP + FN + FP + TN)
Kappa = (obs_agree - expec_agree) / (1 - expec_agree)

TP_rate = TP/(TP+FN)
FN_rate = FN/(TP+FN)
TN_rate = TN/(TN+FP)
FP_rate = FP/(TN+FP)

sensitivity = TP /(TP + FN)
# specificity, selectivity or true negative rate (TNR)
specificity = TN/(TN+FP)

cdf_sns = pd.DataFrame({"name": [], "BN":[],"TP": [],"TN":[],"FP":[],"FN":[],"Metrics_value":[],"Metric_name":[]})
df_ACC = pd.DataFrame({"name":name,"BN": bn, "TN": TN,"TP": TP,"FP":FP,"FN": FN,"Metrics_value": ACC, "Metric_name":"ACC"})
df_NPV = pd.DataFrame({"name":name,"BN": bn, "TN": TN,"TP": TP,"FP":FP,"FN": FN,"Metrics_value": NPV, "Metric_name":"NPV"})
df_FPR = pd.DataFrame({"name":name,"BN": bn, "TN": TN,"TP": TP,"FP":FP,"FN": FN,"Metrics_value": FPR, "Metric_name":"FPR"})
df_FOR = pd.DataFrame({"name":name,"BN": bn, "TN": TN,"TP": TP,"FP":FP,"FN": FN,"Metrics_value": FOR, "Metric_name":"FOR"})
df_FNR = pd.DataFrame({"name":name,"BN": bn, "TN": TN,"TP": TP,"FP":FP,"FN": FN,"Metrics_value": FNR, "Metric_name":"FNR"})
df_FDR = pd.DataFrame({"name":name,"BN": bn, "TN": TN,"TP": TP,"FP":FP,"FN": FN,"Metrics_value": FDR, "Metric_name":"FDR"})
df_precision = pd.DataFrame({"name":name,"BN": bn, "TN": TN,"TP": TP,"FP":FP,"FN": FN,"Metrics_value": precision, "Metric_name":"precision"})
df_specificity = pd.DataFrame({"name":name,"BN": bn, "TN": TN,"TP": TP,"FP":FP,"FN": FN,"Metrics_value": specificity, "Metric_name":"specificity"})
df_sensitivity = pd.DataFrame({"name":name,"BN": bn, "TN": TN,"TP": TP,"FP":FP,"FN": FN,"Metrics_value": sensitivity, "Metric_name":"sensitivity"})
df_Kappa = pd.DataFrame({"name":name,"BN": bn, "TN": TN,"TP": TP,"FP":FP,"FN": FN,"Metrics_value": Kappa, "Metric_name":"Kappa"})
df_FM = pd.DataFrame({"name":name,"BN": bn, "TN": TN,"TP": TP,"FP":FP,"FN": FN,"Metrics_value": FM, "Metric_name":"FM"})
dice_coef = pd.DataFrame({"name":name,"BN": bn, "TN": TN,"TP": TP,"FP":FP,"FN": FN,"Metrics_value": dice_coef, "Metric_name":"dice_coef"})


df_TP_rate = pd.DataFrame({"name":name,"BN": bn, "TN": TN,"TP": TP,"FP":FP,"FN": FN,"Metrics_value": TP_rate, "Metric_name":"TP_rate"})
df_FN_rate = pd.DataFrame({"name":name,"BN": bn, "TN": TN,"TP": TP,"FP":FP,"FN": FN,"Metrics_value": FN_rate, "Metric_name":"FN_rate"})
df_TN_rate= pd.DataFrame({"name":name,"BN": bn, "TN": TN,"TP": TP,"FP":FP,"FN": FN,"Metrics_value": TN_rate, "Metric_name":"TN_rate"})
df_FP_rate = pd.DataFrame({"name":name,"BN": bn, "TN": TN,"TP": TP,"FP":FP,"FN": FN,"Metrics_value": FP_rate, "Metric_name":"FP_rate"})
data = [df_TP_rate["Metrics_value"]]
data[0].dropna()
# print(len(bn))
df = pd.DataFrame({"name":name,"BN": bn, "TN": TN,"TP": TP,"FP":FP,"FN": FN})
# boxplot = df_TP_rate.boxplot(column=["Metrics_value"])
# plt.show()
fig2, ax2 = plt.subplots()
# ax2.set_title('Notched boxes')
ax2.boxplot(data[0])
plt.show()
# Make the figure
red_diamond = dict(markerfacecolor='r', marker='D')
# Create the boxplot and store the resulting python dictionary
my_boxes = ax2.boxplot(df_TP_rate["Metrics_value"], notch=True)

# Call the function to make labels
make_labels(ax2, my_boxes)
plt.show()
cdf = pd.concat([ACC, sensitivity, specificity, precision, FPR, NPV, FNR, FOR], axis=1)
cdf_sns = pd.concat([df_ACC, df_sensitivity, df_specificity], axis=0)
std_val = df_FP_rate['Metrics_value'].std(axis = 0, skipna = True)
mean_val = df_FP_rate['Metrics_value'].median()
q1 = mean_val+ std_val
cdf_new = pd.concat([df_TP_rate,df_TN_rate,df_FP_rate,df_FN_rate], axis=0)
df_FP_rate.loc[(df_FP_rate['Metrics_value'] > 0.5) & (df_FP_rate['BN'] == "with_BN")].to_csv('df_FP_with_bn.csv')
df_FP_rate.loc[(df_FP_rate['Metrics_value'] > 0.5) & (df_FP_rate['BN'] == "with_out_BN")].to_csv('df_FP_with_out_bn.csv')
df_TP_rate.loc[(df_TP_rate['Metrics_value'] < 0.5) & (df_TP_rate['BN'] == "with_BN")].to_csv('df_TP_with_bn.csv')
df_TP_rate.loc[(df_TP_rate['Metrics_value'] < 0.5) & (df_TP_rate['BN'] == "with_out_BN")].to_csv('df_TP_with_out_bn.csv')
df_FP_rate.to_csv('df_FP_rate.csv')
cdf.to_csv("E:\\camelyon16\\dataset_for_training\\10k_dataset\\10k_level0_balanced\\" + 'all_metrics.csv')
cdf_sns.to_csv("E:\\camelyon16\\dataset_for_training\\10k_dataset\\10k_level0_balanced\\" + 'all_metrics_cdf_sns.csv')
# cdf_new.to_csv('all_metrics.csv')
# cdf_sns = cdf_sns.replace(0, np.NaN)
# cdf_sns.dropna()
g = sns.catplot(hue="BN", y="Metrics_value",
                x="Metric_name",
                data=cdf_new, kind= "box",palette="Oranges_r");
plt.show()
g = sns.catplot(hue="BN", y="Metrics_value",
                x="Metric_name",
                data=cdf_sns, kind= "box",palette="Oranges_r");

# g = sns.catplot(hue="BN", y="Metrics_value",
#                 x="Metric_name",
#                 data=cdf_kappa, kind= "box",palette="Oranges_r");
# boxplot = cdf.boxplot(column=["ACC", "sensitivity", "specificity", "precision", "FPR", "NPV", "FNR", "FOR"])
plt.show()

#
# df_2 = pd.DataFrame(df_2, columns=["FP_per","FN_per", "TP_per", "TN_per"])
# boxplot = df_2.boxplot(column=[ "FP_per","FN_per","TP_per", "TN_per"])
# plt.show()
# # print(cdf.head())
# df_1 = pd.DataFrame(df_1, columns=["FP_per","FN_per", "TP_per", "TN_per"])
# boxplot = df_1.boxplot(column=[ "FP_per","FN_per","TP_per", "TN_per"])
# plt.show()
# plt.savefig("boxplot1.png")
# df_2 = pd.DataFrame(df_2, columns=["FP_per","FN_per", "TP_per", "TN_per"])
# boxplot = df_2.boxplot(column=[ "FP_per","FN_per","TP_per", "TN_per"])
# plt.show()
# plt.savefig("boxplot2.png")
# df = pd.DataFrame(df,
#                   columns=['TN', 'TP', 'FN', 'FP'])
# boxplot = df.boxplot(column=['TN', 'TP', 'FN', 'FP'])


# plt.show()