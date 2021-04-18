import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os.path as osp
import openslide
from pathlib import Path

from skimage.filters import threshold_otsu
import glob
import cv2 as cv2
from skimage import io
import xml.etree.ElementTree as et
import pandas as pd
import math


print('Hi, patch extraction can take a while, please be patient...')
slide_path = 'E:/camelyon16/TrainingData/tumor/'
# slide_path_normal = 'E:/camelyon16/TrainingData/normal/'

anno_path = 'E:/camelyon16/TrainingData/lesion_annotations/'
BASE_TRUTH_DIR = 'E:/camelyon16/TrainingData/Ground_truth/mask/'
patches_path = 'E:/camelyon16/TrainingData/test/patches/imgs/'

slide_paths = glob.glob(osp.join(slide_path, '*.tif'))
slide_paths.sort()
# slide_paths_normal = glob.glob(osp.join(slide_path_normal, '*.tif'))
# slide_paths_normal.sort()
slide_paths_total = slide_paths
#slide_paths_total = slide_paths + slide_paths_normal
BASE_TRUTH_DIRS = glob.glob(osp.join(BASE_TRUTH_DIR, '*.tif'))
Anno_paths = glob.glob(osp.join(anno_path, '*.xml'))
BASE_TRUTH_DIRS.sort()

def is_sorta_tumor(arr, threshold=0.5):
    tot = np.float(np.sum(arr))
    if tot/arr.size < (threshold):
       # print ("is not black" )
       return False
    elif (tot/arr.size == 1):
        return False
    else:
        return True

def convert_xml_df(file):
    parseXML = et.parse(file)
    root = parseXML.getroot()
    dfcols = ['Name', 'Order', 'X', 'Y']
    df_xml = pd.DataFrame(columns=dfcols)
    for child in root.iter('Annotation'):
        for coordinate in child.iter('Coordinate'):
            Name = child.attrib.get('Name')
            Order = coordinate.attrib.get('Order')
            X_coord = float(coordinate.attrib.get('X'))
            # X_coord = X_coord - 30000
            # X_coord = ((X_coord)*dims[0])/Ximageorg
            Y_coord = float(coordinate.attrib.get('Y'))
            # Y_coord = Y_coord - 155000
            # Y_coord = ((Y_coord)*dims[1])/Yimageorg
            df_xml = df_xml.append(pd.Series([Name, Order, X_coord, Y_coord], index=dfcols), ignore_index=True)
            df_xml = pd.DataFrame(df_xml)
    return (df_xml)


def random_crop(slide, truth, thresh, crop_size, bbox):
    # width, height = slide.level_dimensions[0]
    # xxmin = list(bboxt['xmin'].get_values())
    # xxmax = list(bboxt['xmax'].get_values())
    # yymin = list(bboxt['ymin'].get_values())
    # yymax = list(bboxt['ymax'].get_values())
    dy, dx = crop_size

    x = bboxt['xmin'], bboxt['xmin'] - dx + 1
    y = bboxt['ymin'], bboxt['ymin'] - dy + 1
    # x = np.random.choice(range(width - dx + 1), replace = False)
    # y = np.random.choice(range(height - dy +1), replace = False)
    index = [x, y]
    # print(index)
    # cropped_img = (image[x:(x+dx), y:(y+dy),:], rgb_binary[x:(x+dx), y:(y+dy)], mask[x:(x+dx), y:(y+dy)])
    rgb_image = slide.read_region((x, y), 0, crop_size)
    rgb_mask = truth.read_region((x, y), 0, crop_size)
    rgb_mask = (cv2.cvtColor(np.array(rgb_mask), cv2.COLOR_RGB2GRAY) > 0).astype(int)
    rgb_array = np.array(rgb_image)
    hsv_rgbimage = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2HSV)
    rgb_binary = cv2.inRange(hsv_rgbimage, thresh[0], thresh[1])
    # cropped_img = image[x:(x+dx), y:(y+dy),:]
    # cropped_binary = rgb_binary[x:(x+dx), y:(y+dy)]
    # cropped_mask = mask[x:(x+dx), y:(y+dy)]
    # print(index)
    return (rgb_image, rgb_binary, rgb_mask, index)


# sampletotal = pd.DataFrame([])
crop_size = [256, 256]
i = 0
while i < len(slide_paths):
    # sampletotal = pd.DataFrame([])
    base_truth_dir = Path(BASE_TRUTH_DIR)
    anno_path = Path(anno_path)
    slide_contains_tumor = osp.basename(slide_paths_total[i]).startswith('tumor_')

    with openslide.open_slide(slide_paths_total[i]) as slide:
        thumbnail = slide.get_thumbnail((slide.dimensions[0] / 256, slide.dimensions[1] / 256))
        thum = np.array(thumbnail)
        hsv_image = cv2.cvtColor(thum, cv2.COLOR_BGR2HSV)
        # cv2.imwrite("hsv_image.png",hsv_image)
        h, s, v = cv2.split(hsv_image)
        hthresh = threshold_otsu(h)
        sthresh = threshold_otsu(s)
        vthresh = threshold_otsu(v)
        # be min value for v can be changed later
        minhsv = np.array([hthresh, sthresh, 70], np.uint8)
        maxhsv = np.array([180, 255, vthresh], np.uint8)
        thresh = [minhsv, maxhsv]
        # extraction the countor for tissue

        rgbbinary = cv2.inRange(hsv_image, thresh[0], thresh[1])
        contours, _ = cv2.findContours(rgbbinary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxtcols = ['xmin', 'xmax', 'ymin', 'ymax']
        bboxt = pd.DataFrame(columns=bboxtcols)
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            bboxt = bboxt.append(pd.Series([x, x + w, y, y + h], index=bboxtcols), ignore_index=True)
            bboxt = pd.DataFrame(bboxt)

        xxmin = list(bboxt['xmin'].get_values())
        xxmax = list(bboxt['xmax'].get_values())
        yymin = list(bboxt['ymin'].get_values())
        yymax = list(bboxt['ymax'].get_values())
        bboxt = math.floor(np.min(xxmin) * 256), math.floor(np.max(xxmax) * 256), math.floor(
            np.min(yymin) * 256), math.floor(np.max(yymax) * 256)


    if slide_contains_tumor:

        truth_slide_path = base_truth_dir / osp.basename(slide_paths_total[i]).replace('.tif', '_mask.tif')
        Anno_pathxml = anno_path / osp.basename(slide_paths_total[i]).replace('.tif', '.xml')
        count = 0
        while (count < 100):
            with openslide.open_slide(str(truth_slide_path)) as truth:

                slide = openslide.open_slide(slide_paths_total[i])
                annotations = convert_xml_df(str(Anno_pathxml))

                x_values = list(annotations['X'].get_values())
                y_values = list(annotations['Y'].get_values())
                bbox = math.floor(np.min(x_values)), math.floor(np.max(x_values)), math.floor(np.min(y_values)), math.floor(
                    np.max(y_values))

                m = 0


                r = random_crop(slide, truth, thresh, crop_size, bbox)
                # if (is_sorta_tumor(r[2])==True):
                imageio.imwrite(patches_path+ '/%s_%d_%d.png' % (
                osp.splitext(osp.basename(slide_paths_total[i]))[0], r[3][0], r[3][1]), r[0])

                imageio.imwrite(patches_path + 'masks/%s_%d_%d_mask.png' % (
                osp.splitext(osp.basename(slide_paths_total[i]))[0], r[3][0], r[3][1]), r[2])
                m = m + 1
                count = count + 1

        i = i+1
