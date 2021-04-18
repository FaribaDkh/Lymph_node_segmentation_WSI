import csv
import os
# from os import path
# cv2 package is installed using pip install opencv-python
import cv2
import seaborn as sns
# current path is fetched
current_path = os.getcwd()
import numpy as np
import pandas as pd
import sys
import imutils
sys.setrecursionlimit(1500)
# folder is fetched where images are located dynamically
path = "E:\\newTumorPatcheDetection\\"
dataset_type = "config_TUPAC16"
# folderReal = os.path.join(path, 'real')
# folderFake = os.path.join(path, 'fake')
folder_pred = os.path.join(path, 'preds_mask')
folder_true = os.path.join(path, 'true_mask')

# loop us implemented to read the images one by one from the loop.
# listdir return a list containing the names of the entries
# in the directory.
itr = 0
arraysSinglPixel = []
arraysSinglIouOne = []
arraysSinglIouTwo = []
cmTotal = []
f1Tot = []
def bound(tuple, low=0, high=2000):
    xs, ys = tuple
    return min(max(xs, low), high), min(max(ys, low), high)
#floodfill on image
def floodfill(bin_map, x, y, visited, grid_markoff):
    if len(visited) > 1000:
        return
    if x >= grid_markoff.shape[0] or y >= grid_markoff.shape[1] or x < 0 or y < 0:
        return
    if grid_markoff[x, y]:
        return
    if not bin_map[x, y]:
        return
    grid_markoff[x, y] = True
    visited.append((x, y))
    floodfill(bin_map, x+1, y, visited, grid_markoff)
    floodfill(bin_map, x, y+1, visited, grid_markoff)
    floodfill(bin_map, x-1, y, visited, grid_markoff)
    floodfill(bin_map, x, y-1, visited, grid_markoff)
MIN_DIST = 25.0
MIN_SIZE = 20
#gets nuclei number, avg nuclei statistics
def centroid_coords(heatmap):
    #IPython.embed()
    grid_markoff = np.zeros(shape=heatmap.shape)
    coords = []
    num_nuclei = 0
    avg_nuclei = 0
    for x in range(0, heatmap.shape[0]):
        for y in range(0, heatmap.shape[1]):
            visited = []
            floodfill(heatmap, x, y, visited, grid_markoff)
            if len(visited) > 0:
                # print ("Found nuclei!", x, y)
                num_nuclei += 1
                avg_nuclei += len(visited)
                centroid_nuclei = np.array(visited).mean(axis=0)
                if len(visited) < MIN_SIZE:
                    continue
                # print (len(visited), centroid_nuclei)
                coords.append((centroid_nuclei[0], centroid_nuclei[1]))
    return coords
from scipy.linalg import norm

def remove_small_particles(greenPixels):
    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(greenPixels, connectivity=8)
    # connectedComponentswithStats yields every seperated component with information on each of them, such as size
    # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1];
    nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 50

    # your answer image
    img2 = np.zeros((output.shape))
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    kernel = np.ones((2, 2), np.uint8)
    # img2 = cv2.dilate(img2, kernel, iterations=1)
    # img2 = cv2.dilate(img2, kernel, iterations=1)
    img2 = cv2.dilate(img2, kernel, iterations=1)
    img2 = cv2.dilate(img2, kernel, iterations=1)
    return img2

pred_heatmaps, true_maps = [], []
image_list = []
f1_per_patch, tp, num_pred, num_true, = 0, 0, 0, 0
df = pd.DataFrame(
    {"file_name": [], "tp_per_patch": [], "num_pred_per_patch": [], "num_true_per_patch": [], "f1_per_patch": [],
     "precision": [],
     "recall": [],"type":"ICPR12"})
kernel = np.ones((3, 3), np.uint8)
for filename in os.listdir(folder_true):
    img_real = cv2.imread((os.path.join(folder_true, filename)),0)
    imgFake = cv2.imread((os.path.join(folder_pred, filename)),0)

    # img_real = cv2.dilate(img_real, kernel, iterations=1)
    # img_real = cv2.dilate(img_real, kernel, iterations=1)
    # img_real = cv2.dilate(img_real, kernel, iterations=1)
    img_real = cv2.dilate(img_real, kernel, iterations=1)
    cv2.imwrite("E:\\newTumorPatcheDetection\\dilated_true\\" + filename, img_real)
    # img_real = cv2.dilate(img_real, iterations=1)
    # img_real = cv2.dilate(img_real, iterations=1)
    # imgFake = remove_small_particles(imgFake)
    # calculate moments of binary image
    # imgFake = np.uint8(imgFake)

    # grayscale_image = cv2.cvtColor(image1copy, cv2.COLOR_HSV2BGR)
    nb_components_pred, output_pred, stats_pred, centroids_pred = cv2.connectedComponentsWithStats(imgFake, connectivity=8)
    nb_components_real, output_real, stats_real, centroids_real = cv2.connectedComponentsWithStats(img_real, connectivity=8)
    print("centroids_pred: ",centroids_pred)
    print("centroids_real: ",centroids_real)
    tp_per_patch, num_pred_per_patch, num_true_per_patch, precision_per_patch, recall_per_patch = 0, 0, 0, 0, 0
    list_centroids_preds = centroids_pred[1:]
    list_centroids_true = centroids_real[1:]
    # centroids_pred = centroid_coords(imgFake)
    # centroids_true = centroid_coords(imgReal)
    num_pred += len(list_centroids_preds)
    num_true += len(list_centroids_true)
    num_pred_per_patch = len(list_centroids_preds)
    num_true_per_patch = len(list_centroids_true)

    for x2, y2 in list_centroids_true:
        for x1, y1 in list_centroids_preds:
            # print ((x1, y1), (x2, y2), int(norm(((x1 - y1), (x2 - y2)))))
            if (norm(((x1 - x2), (y1 - y2))) < MIN_DIST):
                # if ((tp_per_patch) < num_true_per_patch):
                tp += 1
                tp_per_patch += 1
                break

    num_pred_per_patch = num_pred_per_patch
    num_true_per_patch = num_true_per_patch
    if (num_pred_per_patch != 0 or recall_per_patch != 0):
        precision_per_patch = tp_per_patch * 1.0 / num_pred_per_patch
        recall_per_patch = tp_per_patch * 1.0 / num_true_per_patch
    if (recall_per_patch != 0 and precision_per_patch != 0):
        F1_per_patch = (2 * precision_per_patch * recall_per_patch) / (precision_per_patch + recall_per_patch)
    df = df.append({"file_name": filename, "tp_per_patch": tp_per_patch, "num_pred_per_patch": num_pred_per_patch,
                    "num_true_per_patch": num_true_per_patch,
                    "f1_per_patch": F1_per_patch, "precision": precision_per_patch, "recall": recall_per_patch,"dataset": dataset_type},
                   ignore_index=True)
    df.to_csv(dataset_type + '.csv')
    #     print (tp)
    # print (tp, num_pred, num_true)
num_pred = max(num_pred, 1)
num_true = max(num_true, 1)
precision = tp * 1.0 / num_pred
recall = tp * 1.0 / num_true
F1_all = (2 * precision * recall) / (precision + recall)
# give precision, recall and F1 as scores.
pred_heatmaps, true_maps = np.array(pred_heatmaps), np.array(true_maps)
# f1score = f1_score_per_patch(pred_heatmaps, true_maps,image_list)
print('F1 score: {}'.format(F1_all))
print('recall score: {}'.format(recall))
print('precision score: {}'.format(precision))