import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os.path as osp
import openslide
from pathlib import Path
import imageio
from skimage.filters import threshold_otsu
import glob
#before importing HDFStore, make sure 'tables' is installed by pip3 install tables
from pandas import HDFStore
from openslide.deepzoom import DeepZoomGenerator
import cv2 as cv

from skimage import io
import xml.etree.ElementTree as et
import pandas as pd
import math
from skimage import color
# from skimage import io
# setup for path
# BASE_TRUTH_DIR = Path('/home/ubuntu/data/Ground_Truth_Extracted/Mask')

# slide_path = '/home/ubuntu/data/slides/Tumor_009.tif'
# truth_path = str(BASE_TRUTH_DIR / 'Tumor_009_Mask.tif')
# BASE_TRUTH_DIR = Path('/Users/liw17/Downloads/camelyontest/Mask')

# slide_path = '/Users/liw17/Downloads/camelyontest/slides/tumor_026.tif'
# truth_path = osp.join(BASE_TRUTH_DIR, 'tumor_026_mask.tif')


# slide = openslide.open_slide(slide_path)
# truth = openslide.open_slide(truth_path)

print('Hi, patch extraction can take a while, please be patient...')
slide_path = 'E:\\CAM17\\ann_slide\\'
anno_path = 'E:\\CAM17\\lesion_annotations\\'
BASE_TRUTH_DIR = 'E:\\CAM17\\mask2017\\'
# slide_path = 'E:\\CAM17\\ann_slide\\'
# anno_path = 'E:\\CAM17\\lesion_annotations\\'
# BASE_TRUTH_DIR = 'E:\\CAM17\\mask2017\\'
# all_tumor_patches = "E:\\camelyon16\\dataset_for_training\\all_tumor_patches_test\\"
# anno_path = '/home/wli/Downloads/CAMELYON16/training/Lesion_annotations'
# BASE_TRUTH_DIR = '/home/wli/Downloads/CAMELYON16/masking'
slide_paths = glob.glob(osp.join(slide_path, '*.tif'))
slide_paths.sort()

# slide_paths_total = slide_paths
slide_paths_total = slide_paths
BASE_TRUTH_DIRS = glob.glob(osp.join(BASE_TRUTH_DIR, '*.tif'))
Anno_paths = glob.glob(osp.join(anno_path, '*.xml'))
BASE_TRUTH_DIRS.sort()

LEVEL = 0

# image_pair = zip(tumor_paths, anno_tumor_paths)
# image_pair = list(image_mask_pair)
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
    dy, dx = crop_size
    x = np.random.randint(bbox[0], bbox[1] - dx + 256)
    y = np.random.randint(bbox[2], bbox[3] - dy + 256)
    # x = np.random.choice(range(width - dx + 1), replace = False)
    # y = np.random.choice(range(height - dy +1), replace = False)
    index = [x, y]
    # print(index)
    # cropped_img = (image[x:(x+dx), y:(y+dy),:], rgb_binary[x:(x+dx), y:(y+dy)], mask[x:(x+dx), y:(y+dy)])
    rgb_image = slide.read_region((x, y), 4, crop_size)
    rgb_mask = truth.read_region((x, y), 4, crop_size)
    rgb_mask = (cv.cvtColor(np.array(rgb_mask), cv.COLOR_RGB2GRAY) > 0).astype(int)
    rgb_array = np.array(rgb_image)
    hsv_rgbimage = cv.cvtColor(rgb_array, cv.COLOR_BGR2HSV)
    rgb_binary = cv.inRange(hsv_rgbimage, thresh[0], thresh[1])
    # cropped_img = image[x:(x+dx), y:(y+dy),:]
    # cropped_binary = rgb_binary[x:(x+dx), y:(y+dy)]
    # cropped_mask = mask[x:(x+dx), y:(y+dy)]
    print(index)
    return (rgb_image, rgb_binary, rgb_mask, index)


def None_overlapped_crop(slide, truth, thresh, crop_size, bbox,j,k):
    # width, height = slide.level_dimensions[0]
    dy, dx = crop_size
    # x = bbox[0] + (k*(size*mag_factor))
    # y = bbox[2] + (j*(size*mag_factor))
    x = bbox[0] + (k*(size*mag_factor))
    y = bbox[2] + (j*(size*mag_factor))
    # x = bbox[0] + (k*(size*mag_factor))
    # y = bbox[2] + (j*(size*mag_factor))
    # x = np.random.randint(bboxt[0], bboxt[1] - dx + 1)
    # y = np.random.randint(bboxt[2], bboxt[3] - dy + 1)
    # x = bbox[0] + (num_column*(size*mag_factor))
    # y = bbox[2] + (num_rows*(size*mag_factor))
    # x = np.random.choice(range(width - dx + 1), replace = False)
    # y = np.random.choice(range(height - dy +1), replace = False)
    index = [x, y]
    # print(index)
    # cropped_img = (image[x:(x+dx), y:(y+dy),:], rgb_binary[x:(x+dx), y:(y+dy)], mask[x:(x+dx), y:(y+dy)])
    rgb_image = slide.read_region((x, y),LEVEL, crop_size)
    rgb_mask = truth.read_region((x, y), LEVEL, crop_size)

    # rgb_image = slide.read_region((x, y), 1, crop_size)
    # rgb_mask = truth.read_region((x, y), 1, crop_size)
    rgb_mask = (cv.cvtColor(np.array(rgb_mask), cv.COLOR_RGB2GRAY) > 0).astype(int)
    rgb_array = np.array(rgb_image)
    hsv_rgbimage = cv.cvtColor(rgb_array, cv.COLOR_BGR2HSV)
    rgb_binary = cv.inRange(hsv_rgbimage, thresh[0], thresh[1])
    # cropped_img = image[x:(x+dx), y:(y+dy),:]
    # cropped_binary = rgb_binary[x:(x+dx), y:(y+dy)]
    # cropped_mask = mask[x:(x+dx), y:(y+dy)]
    # print(index)
    return (rgb_image, rgb_binary, rgb_mask, index)
def testduplicates(list):
    for each in list:
        count = list.count(each)
        if count > 1:
            z = 0
        else:

            z = 1
    return z


# sampletotal = pd.DataFrame([])
crop_size = [256, 256]
size = 256
i = 0
for i in range(len(slide_paths_total)):
    # sampletotal = pd.DataFrame([])
    base_truth_dir = Path(BASE_TRUTH_DIR)
    anno_path = Path(anno_path)
    # slide_contains_tumor = osp.basename(slide_paths_total[i]).startswith('tumor_')

    with openslide.open_slide(slide_paths_total[i]) as slide:
        thumbnail = slide.get_thumbnail((slide.dimensions[0] / 256, slide.dimensions[1] / 256))
        thum = np.array(thumbnail)
        hsv_image = cv.cvtColor(thum, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv_image)
        hthresh = threshold_otsu(h)
        sthresh = threshold_otsu(s)
        vthresh = threshold_otsu(v)
        # be min value for v can be changed later
        minhsv = np.array([hthresh, sthresh, 70], np.uint8)
        maxhsv = np.array([180, 255, vthresh], np.uint8)
        thresh = [minhsv, maxhsv]
        # extraction the countor for tissue
        # cv.imshow("hsv_image",hsv_image)
        gray_image = cv.cvtColor(thum, cv.COLOR_BGR2GRAY)
        ret2, rgbbinary = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # rgbbinary = cv.inRange(hsv_image, thresh[0], thresh[1])
        cv.imwrite("hsv_image.png", rgbbinary)
        contours, _ = cv.findContours(rgbbinary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        bboxtcols = ['xmin', 'xmax', 'ymin', 'ymax']
        bboxt = pd.DataFrame(columns=bboxtcols)
        for c in contours:
            (x, y, w, h) = cv.boundingRect(c)
            bboxt = bboxt.append(pd.Series([x, x + w, y, y + h], index=bboxtcols), ignore_index=True)
            bboxt = pd.DataFrame(bboxt)
        # xxmax = bboxt.T[1][1]+1
        # yymin = bboxt.T[1][2]-1
        # yymax = bboxt.T[0][3]+1
        # list(bboxt['xmin'].values)
        xxmin = list(bboxt['xmin'].values)
        xxmax = list(bboxt['xmax'].values)
        yymin = list(bboxt['ymin'].values)
        yymax = list(bboxt['ymax'].values)
        #
        # print(math.floor(xxmin * 256))
        # print(math.floor(xxmax * 256))
        # print(math.floor(yymin * 256))
        # print(math.floor(yymax * 256))
        # bboxt = xxmin, xxmax, yymin, yymax
        bboxt = math.floor(np.min(xxmin) * 256), math.floor(np.max(xxmax) * 256), math.floor(
            np.min(yymin) * 256), math.floor(np.max(yymax) * 256)

        truth_slide_path = base_truth_dir / osp.basename(slide_paths[i]).replace('.tif', '_mask.tif')
        Anno_pathxml = anno_path / osp.basename(slide_paths[i]).replace('.tif', '.xml')

        with openslide.open_slide(str(truth_slide_path)) as truth:

            slide = openslide.open_slide(slide_paths[i])
            annotations = convert_xml_df(str(Anno_pathxml))
            x_values = list(annotations['X'].values)
            y_values = list(annotations['Y'].values)
            # bbox = math.floor(np.min(x_values)), math.floor(np.max(x_values)), math.floor(np.min(y_values)), math.floor(
            #     np.max(y_values))

            # a = []
            # print(np.min(x_values))
            # print(np.max(x_values))
            # print(np.min(y_values))
            # print(np.max(y_values))
            # print(bbox)
            import os

            filepath, e = os.path.splitext(slide_paths[i])
            filename = osp.splitext(osp.basename(filepath))[0]
            # define the name of the directory to be created
            path = "E:\\camelyon16\\camelyon2017\\" + filename
            path_mask = "E:\\camelyon16\\camelyon2017\\" + filename
            #
            try:
                os.mkdir(path)
            except OSError:
                print("Creation of the directory %s failed" % path)
            else:
                print("Successfully created the directory %s " % path)

            try:
                os.mkdir(path+"/masks")

            except OSError:
                print("Creation of the directory %s failed" % path)
            else:
                print("Successfully created the directory %s " % path)
            try:
                os.mkdir(path+"/patches")
            except OSError:
                print("Creation of the directory %s failed" % path)
            else:
                print("Successfully created the directory %s " % path)
            try:
                os.mkdir(path + "/masks_1")

            except OSError:
                print("Creation of the directory %s failed" % path)
            else:
                print("Successfully created the directory %s " % path)

            # patch_path = "E:\\camelyon16\\dataset_for_training\\10k_dataset\\imgs\\"
            wsi_image = openslide.OpenSlide(slide_paths[i])
            level_used = wsi_image.level_count - 1
            # text_file = open(patch_path+"/" + filename + ".txt", "w")

            level_used = 9 - LEVEL
            mag_factor = pow(2, LEVEL)
            mag_factor2 = pow(2, level_used)
            # print(level_used)
            # bbox = math.floor(np.min(x_values)), math.floor(np.max(x_values)), math.floor(np.min(y_values)), math.floor(
            #     np.max(y_values))
            # bboxt = math.floor(np.min(xxmin) * 256), math.floor(np.max(xxmax) * 256), math.floor(
            #     np.min(yymin) * 256), math.floor(np.max(yymax) * 256)

            ###### just bounding box for tumor area
            # num_column = int((math.floor(np.max(x_values)) - math.floor(np.min(x_values)))/(size*mag_factor)) + 1
            # num_rows = int((math.floor(np.max(y_values)) - math.floor(np.min(y_values)))/(size*mag_factor)) +1

            ##### bounding box for all of the slide


            # num_column = int((math.floor(np.max(xxmax) * 256) - math.floor(np.min(xxmin) * 256))/(size*mag_factor))+(9-LEVEL)
            num_column = int((math.floor(np.max(xxmax)*256) - math.floor(np.min(xxmin)*256))/(size*mag_factor))+1
            num_rows = int((math.floor(np.max(yymax)*256) - math.floor(np.min(yymin)*256))/(size*mag_factor))+1
            print(num_column)
            print(num_rows)
            text_file = open(path + "/" + filename + ".txt", "w")
            text_file.write("Slide name : %s" % filename)
            text_file.write("\n")
            text_file.write("level used : %s" % LEVEL)
            text_file.write("\n")
            text_file.write("num_column : %s" % num_column)
            text_file.write("\n")
            text_file.write("num_rows : %s" % num_rows)
            text_file.write("\n ______________________________")
            text_file.close()

            n = 0
            for k in range(num_rows):
                 for j in range(num_column):

                    # o = 0
                    # while o in range(0, 2):
                    r = None_overlapped_crop(slide, truth, thresh,crop_size,bboxt,k,j)
                    # r = None_overlapped_crop_fariba(slide, truth, thresh,crop_size,bboxt)
                    # if (cv2.countNonZero(r[2]) > crop_size[0] * crop_size[1] * 0.5):
                    # I used r1 for exracting all patchas with tissue
                    # if (cv.countNonZero(r[1]) > crop_size[0] * crop_size[1] * 0.5):
                    # if (cv.countNonZero(r[2]) ==0):
                    img = color.rgba2rgb(r[0])
                    # io.imsave(path+'/masks/'+str(n)+'.png', r[2])
                    mask = r[2] / 255
                    mask = color.gray2rgb(mask)
                    # io.imsave(patch_path+'/masks_1/'+str(n)+'.png', mask)
                    # print(r[2])
                    if(cv.countNonZero(r[1]) > crop_size[0] * crop_size[1] * 0.01 or cv.countNonZero(r[2])> 10 ):
                        io.imsave(path + "/masks_1" +'/%s.png' % (str(n)), mask*255)
                        io.imsave(path + "/patches"+'/%s.png' % (str(n)),img)
                        io.imsave(path + "/masks"+'/%s.png' % (str(n)), mask)
                        # o = o + 1
                    n+=1
                    # print(r[2])
            # while m in range(0, 10):
                # r = random_crop(slide, truth, thresh, crop_size, bbox)
                # if (cv2.countNonZero(r[2]) > crop_size[0] * crop_size[1] * 0.5) and (m <= 10):
                #     saveim('E:/camelyon16/level4/tumor/%s_%d_%d.png' % (
                #         osp.splitext(osp.basename(slide_paths[i]))[0], r[3][0], r[3][1]), r[0])
                #
                #     io.imsave('E:/camelyon16/level4/mask/%s_%d_%d_mask.png' % (
                #         osp.splitext(osp.basename(slide_paths[i]))[0], r[3][0], r[3][1]), r[2])
                #
                #     print(r[2])
                #
                #     # a.append(r[3])
                #     # z = testduplicates(a)
                #     m = m + 1
                #
                # # else:
                # #     m = m
