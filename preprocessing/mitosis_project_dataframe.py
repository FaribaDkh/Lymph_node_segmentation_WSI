import pandas as pd
import os
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


patient_path = "C:\\Users\\fdamband\\Downloads\\mitoses_ground_truth\\"
patient_name_dir = os.listdir(patient_path)
df = pd.DataFrame({"image_name": [],"center_name": [],"patient_name": [], "HPF": [], "number_of_mitosis": []})
df_per_patient = pd.DataFrame({"patient_name": [], "number_of_mitosis_per_patient": []})
df_mitosis_per_pixel_each_patient = pd.DataFrame({"patient_name": [], "mitosis_per_pixel_each_patient": []})

all_mitosis_of_patient = 0
row_count = 0
HPF_count = 0
mitosis_per_pixel_each_patient = 0

for item in patient_name_dir:
    patient_name, e = os.path.splitext(item)
    HPFz = patient_path + patient_name + "\\"
    HPFz_dir = os.listdir(HPFz)
    for item2 in HPFz_dir:
        # for calculating mitosis_per_pixel_each_patient we need number of HPF for each patient
        HPF_count = HPF_count + 1
        HPFz_name, e = os.path.splitext(item2)
        #  HPF path is defined below
        HPFz_name_csv = HPFz + HPFz_name
        # counting lines of each csv files which shows number of mitosis
        with open(HPFz_name_csv + ".csv", "r") as f:
            reader = csv.reader(f, delimiter=",")
            data = list(reader)
            row_count = len(data)
            all_mitosis_of_patient = row_count + all_mitosis_of_patient
        # Separating the patient based on their centers
        # Define their image size as the image size in center one is 2000*2000 and center two and three are 5657*5657
        if int(patient_name) < 24:
            center_name = "center_one"
            center = 1
            image_size = 2000 * 2000
        elif ((int(patient_name)) > 23) & ((int(patient_name)) < 48):
            center_name = "center_two"
            center = 2
            image_size = 5657 * 5657
        else:
            center_name = "center_three"
            center = 3
            image_size = 5657 * 5657

        # Naming convention applied to each image
        image_name = "Tupac_ROI_Training_" + "center"+str(center) + "_patient" + patient_name + "_HPF" + HPFz_name
        # Append details to dataframe
        df = df.append({"image_name":image_name,"center_name": center_name, "patient_name": item, "HPF": int(HPFz_name), "number_of_mitosis": row_count}, ignore_index=True)
        row_count = 0
    # Analyzing number of mitosis per pixel in whole dataset
    if HPF_count != 0:
        mitosis_per_pixel_each_patient = float(all_mitosis_of_patient / (HPF_count * image_size)) * 10 ** 7
    HPF_count = 0
    df_per_patient = df_per_patient.append({"patient_name": item, "number_of_mitosis_per_patient": all_mitosis_of_patient}, ignore_index=True)
    df_mitosis_per_pixel_each_patient = df_mitosis_per_pixel_each_patient.append({"patient_name": item, "mitosis_per_pixel_each_patient( x10^7)": mitosis_per_pixel_each_patient}, ignore_index=True)
    all_mitosis_of_patient = 0

# write to the csv file
df.to_csv('file_name.csv')
df_per_patient.to_csv('file_name_mitosis_per_patient.csv')
df_mitosis_per_pixel_each_patient.to_csv('mitosis_per_pixel_each_patient.csv')
df.head()
sns.set(style="whitegrid")

# plotting parts
ax = sns.boxplot(y=df_mitosis_per_pixel_each_patient["mitosis_per_pixel_each_patient( x10^7)"], palette="Set3")
plt.show()
f = sns.factorplot(data= df,
               x= 'patient_name',
               y= 'number_of_mitosis',
               hue= 'center_name',kind="bar")

plt.show()

f = sns.factorplot(data= df,
               x= 'center_name',kind="count")
f.set_axis_labels('center_name', 'HPF_count')
plt.show()


