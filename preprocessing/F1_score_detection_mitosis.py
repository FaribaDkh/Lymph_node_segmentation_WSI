import os
# from os import path
# cv2 package is installed using pip install opencv-python
import cv2
# current path is fetched
current_path = os.getcwd()
import numpy as np
# folder is fetched where images are located dynamically
folderReal = os.path.join(current_path, 'real')
folderFake = os.path.join(current_path, 'fake')

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


MIN_DIST = 50.0
MIN_SIZE = 50

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
                print ("Found nuclei!", x, y)
                num_nuclei += 1
                avg_nuclei += len(visited)
                centroid_nuclei = np.array(visited).mean(axis=0)

                if len(visited) < MIN_SIZE:
                    continue

                print (len(visited), centroid_nuclei)
                coords.append((centroid_nuclei[0], centroid_nuclei[1]))

    return coords


from scipy.linalg import norm


def f1_score(heatmaps, true_maps):

    tp, num_pred, num_true = 0, 0, 0
    for i in range(0, len(heatmaps)):
        heatmap, true_map = heatmaps[i], true_maps[i]

        centroids_pred = centroid_coords(heatmap)
        centroids_true = centroid_coords(true_map)
        num_pred += len(centroids_pred)
        num_true += len(centroids_true)

        print (num_pred)
        print (num_true)


        for x2, y2 in centroids_true:
            for x1, y1 in centroids_pred:
                #print ((x1, y1), (x2, y2), int(norm(((x1 - y1), (x2 - y2)))))
                if (norm(((x1 - x2), (y1 - y2))) < MIN_DIST):
                    tp +=1
                    break

        print (tp)

    print (tp, num_pred, num_true)
    num_pred = max(num_pred, 1)
    num_true = max(num_true, 1)
    precision = tp*1.0/num_pred
    recall = tp*1.0/num_true

    return (2*precision*recall)/(precision + recall)



pred_heatmaps, true_maps = [], []
for filename in os.listdir(folderReal):
    imgReal = cv2.imread((os.path.join(folderReal, filename)))
    imgFake = cv2.imread((os.path.join(folderFake, filename)))
    hsv = cv2.cvtColor(imgFake, cv2.COLOR_BGR2HSV)
    mask_1 = cv2.inRange(hsv, (36, 25, 25), (70, 255, 255))
    mask_2 = cv2.inRange(hsv, (0, 50, 20), (5, 255, 255))
    mask_3 = cv2.inRange(hsv, (175, 50, 20), (180, 255, 255))
    #
    # ret1, th1 = cv2.threshold(mask_1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # ret3, th3 = cv2.threshold((mask_2 + mask_3), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # greenPixels = cv2.merge((mask_1, mask_1, mask_1))
    # greenPixels = cv2.cvtColor(greenPixels, cv2.COLOR_HSV2BGR)
    # dd = cv2.cvtColor(mask_2+mask_3, cv2.COLOR_HSV2BGR)
    # mask_123 = cv2.merge((mask_1, mask_1, mask_1))
    # mask_1234 = cv2.merge((mask_2+mask_3, mask_2+mask_3, mask_2+mask_3))
    # cv2.imwrite(filename + "_green_threshold.png", th1)
    # cv2.imwrite(filename + "_red_threshold.png", th3)
    # cv2.imwrite(filename + "_fake_green.png", mask_1)
    # cv2.imwrite(filename + "_fake_red.png", mask_1234)
    # cv2.imwrite(filename + "_original.png", imgReal)
    # cv2.imwrite(filename + "_predicted.png", imgFake)
    # cv2.imwrite(filename + "_original_green.png", imgReal[:, :, 1])
    # cv2.imwrite(filename + "_original_red.png", imgReal[:, :, 2])




    pred_heatmaps.append(imgReal[:, :, 1]/255.0)
    true_maps.append(mask_1/255.0)

pred_heatmaps, true_maps = np.array(pred_heatmaps), np.array(true_maps)

f1score = f1_score(pred_heatmaps, true_maps)


print('F1 score: {}'.format(f1score))