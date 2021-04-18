import sys, os
import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os.path as osp
import openslide
from pathlib import Path
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.filters import threshold_otsu
import glob
#before importing HDFStore, make sure 'tables' is installed by pip3 install tables
from pandas import HDFStore
from openslide.deepzoom import DeepZoomGenerator
import cv2 as cv2
from skimage import io
import xml.etree.ElementTree as et
import pandas as pd
import math


#setup for path 
#BASE_TRUTH_DIR = Path('/home/ubuntu/data/Ground_Truth_Extracted/Mask')

#slide_path = '/home/ubuntu/data/slides/Tumor_009.tif'
#truth_path = str(BASE_TRUTH_DIR / 'Tumor_009_Mask.tif')
#BASE_TRUTH_DIR = Path('/Users/liw17/Downloads/camelyontest/Mask')

#slide_path = '/Users/liw17/Downloads/camelyontest/slides/tumor_026.tif'
#truth_path = osp.join(BASE_TRUTH_DIR, 'tumor_026_mask.tif')


#slide = openslide.open_slide(slide_path)
#truth = openslide.open_slide(truth_path)
print('Hi, patch extraction can take a while, please be patient...')
slide_path = 'E:\\camelyon16\\TrainingData\\tumor\\'
# slide_path_normal = 'E:/camelyon16/TrainingData/normal/'

anno_path = 'E:\\camelyon16\\TrainingData\\lesion_annotations\\'
BASE_TRUTH_DIR = 'E:\\camelyon16\\TrainingData\\Ground_truth\\mask\\'
slide_paths = glob.glob(osp.join(slide_path, '*.tif'))
slide_paths.sort()
# slide_paths_normal = glob.glob(osp.join(slide_path_normal, '*.tif'))
# slide_paths_normal.sort()
slide_paths_total = slide_paths
#slide_paths_total = slide_paths + slide_paths_normal
BASE_TRUTH_DIRS = glob.glob(osp.join(BASE_TRUTH_DIR, '*.tif'))
Anno_paths = glob.glob(osp.join(anno_path, '*.xml'))
BASE_TRUTH_DIRS.sort()

#image_pair = zip(tumor_paths, anno_tumor_paths)  
#image_pair = list(image_mask_pair)

#for color normalization
# go through all the file
def is_sorta_black(arr, threshold=0.8):
    tot = np.float(np.sum(arr))
    # if ((tot/arr.size > (1-threshold)) | ((tot/arr.size) == 1)):
    if (tot / arr.size > (1 - threshold)):
       # print ("is not black" )
       return False
    else:
       # print ("is kinda black")
       return True

def convert_xml_df (file):
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
            #X_coord = ((X_coord)*dims[0])/Ximageorg
            Y_coord = float(coordinate.attrib.get('Y'))
           # Y_coord = Y_coord - 155000
            #Y_coord = ((Y_coord)*dims[1])/Yimageorg
            df_xml = df_xml.append(pd.Series([Name, Order, X_coord, Y_coord], index = dfcols), ignore_index=True)
            df_xml = pd.DataFrame(df_xml)
    return (df_xml)

def random_crop(slide, truth, thresh, crop_size, i,j):
    
    #width, height = slide.level_dimensions[0]
    dy, dx = crop_size
    # x = np.random.randint(bbox[0], bbox[1] - dx + 1)
    # y = np.random.randint(bbox[2], bbox[3] - dy + 1)
    x =  (j * 512) + int(math.floor(np.min(xxmax))* 512)
    y =  (i * 512) + int(math.floor(np.min(yymax))* 512)
    #x = np.random.choice(range(width - dx + 1), replace = False)
    #y = np.random.choice(range(height - dy +1), replace = False)
    index=[x, y]
    #print(index)
    #cropped_img = (image[x:(x+dx), y:(y+dy),:], rgb_binary[x:(x+dx), y:(y+dy)], mask[x:(x+dx), y:(y+dy)])
    rgb_image = slide.read_region((x, y), 0, crop_size)
    print(rgb_image)

    rgb_mask = truth.read_region((x, y), 0, crop_size)
    rgb_mask = (cv2.cvtColor(np.array(rgb_mask), cv2.COLOR_RGB2GRAY) > 0).astype(int)
    rgb_array = np.array(rgb_image)
    hsv_rgbimage = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2HSV)
    rgb_binary = cv2.inRange(hsv_rgbimage, thresh[0], thresh[1])
    #cropped_img = image[x:(x+dx), y:(y+dy),:]
    #cropped_binary = rgb_binary[x:(x+dx), y:(y+dy)]
    #cropped_mask = mask[x:(x+dx), y:(y+dy)]
    # print(rgb_mask)
    # rgb_image = rgb_image.resize((256,256))
    # rgb_mask = rgb_mask.resize((256,256))
    # rgb_binary = rgb_binary.resize((256,256))

    scale_percent = 50  # percent of original size

    return (rgb_image, rgb_binary, rgb_mask, index)

def random_crop2(slide, truth, thresh, crop_size, bboxt):
    
    #width, height = slide.level_dimensions[0]
    dy, dx = crop_size
    x = np.random.randint(bboxt[0], bboxt[1] - dx + 1)
    y = np.random.randint(bboxt[2], bboxt[3] - dy + 1)
    #x = np.random.choice(range(width - dx + 1), replace = False)
    #y = np.random.choice(range(height - dy +1), replace = False)
    index=[x, y]
    #print(index)
    #cropped_img = (image[x:(x+dx), y:(y+dy),:], rgb_binary[x:(x+dx), y:(y+dy)], mask[x:(x+dx), y:(y+dy)])
    rgb_image = slide.read_region((x, y), 1, crop_size)
    rgb_mask = truth.read_region((x, y), 1, crop_size)
    rgb_mask = (cv2.cvtColor(np.array(rgb_mask), cv2.COLOR_RGB2GRAY) > 0).astype(int)
    rgb_array = np.array(rgb_image)
    hsv_rgbimage = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2HSV)
    rgb_binary = cv2.inRange(hsv_rgbimage, thresh[0], thresh[1])
    #cropped_img = image[x:(x+dx), y:(y+dy),:]
    #cropped_binary = rgb_binary[x:(x+dx), y:(y+dy)]
    #cropped_mask = mask[x:(x+dx), y:(y+dy)]
    # print(index)
    # rgb_image = rgb_image.resize((256, 256))
    return (rgb_image, rgb_binary, rgb_mask, index)

def random_crop_normal(slide, thresh, crop_size, bboxt):
    
    #width, height = slide.level_dimensions[0]
    dy, dx = crop_size

    x = np.random.randint(bboxt[0], bboxt[1] - dx + 1)
    y = np.random.randint(bboxt[2], bboxt[3] - dy + 1)
    index=[x, y]

    #cropped_img = (image[x:(x+dx), y:(y+dy),:], rgb_binary[x:(x+dx), y:(y+dy)], mask[x:(x+dx), y:(y+dy)])
    rgb_image = slide.read_region((x, y), 0, crop_size)

    print(rgb_image)
    #rgb_mask = truth.read_region((x, y), 0, crop_size)
    #rgb_mask = (cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2GRAY) > 0).astype(int)
    #rgb_grey = np.array(rgb_image.convert('L'))
    #rgb_binary = (rgb_grey < thresh).astype(int)
    rgb_array = np.array(rgb_image)

    hsv_rgbimage = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2HSV)
    rgb_binary = cv2.inRange(hsv_rgbimage, thresh[0], thresh[1])
    #cropped_img = image[x:(x+dx), y:(y+dy),:]
    #cropped_binary = rgb_binary[x:(x+dx), y:(y+dy)]
    #cropped_mask = mask[x:(x+dx), y:(y+dy)]
    # rgb_image = rgb_image.resize((256, 256))
    return (rgb_image, rgb_binary, index)


def testduplicates(list):
    for each in list:
        count = list.count(each)
        if count > 1:
            z = 0
        else:

            z = 1
    return z

#sampletotal = pd.DataFrame([])
crop_size = [256, 256]
i=0
n = 0
while i < len(slide_paths):
    #sampletotal = pd.DataFrame([])
    base_truth_dir = Path(BASE_TRUTH_DIR)
    anno_path = Path(anno_path)
    # slide_contains_tumor = osp.basename(slide_paths[i]).startswith('tumor_')
    
    with openslide.open_slide(slide_paths[i]) as slide:
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
        #extraction the countor for tissue

        rgbbinary = cv2.inRange(hsv_image, thresh[0], thresh[1])
        contours,_ = cv2.findContours(rgbbinary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxtcols = ['xmin', 'xmax', 'ymin', 'ymax']
        bboxt = pd.DataFrame(columns=bboxtcols)
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            bboxt = bboxt.append(pd.Series([x, x+w, y, y+h], index = bboxtcols), ignore_index=True)
            bboxt = pd.DataFrame(bboxt)
            
        xxmin = list(bboxt['xmin'].get_values())
        xxmax = list(bboxt['xmax'].get_values())
        yymin = list(bboxt['ymin'].get_values())
        yymax = list(bboxt['ymax'].get_values())
        bboxt = math.floor(np.min(xxmin)*256), math.floor(np.max(xxmax)*256), math.floor(np.min(yymin)*256), math.floor(np.max(yymax)*256)

       # thumbnail_grey = np.array(thumbnail.convert('L')) # convert to grayscale
        #thresh = threshold_otsu(thumbnail_grey)
        #binary = thumbnail_grey > thresh
    
        #patches = pd.DataFrame(pd.DataFrame(binary).stack())
        #patches['is_tissue'] = ~patches[0]
        #patches.drop(0, axis=1, inplace=True)
        #patches['slide_path'] = slide_paths[i]
        #patches['threshold']=thresh
    
    # if slide_contains_tumor:
        truth_slide_path = base_truth_dir / osp.basename(slide_paths_total[i]).replace('.tif', '_mask.tif')
        Anno_pathxml = anno_path / osp.basename(slide_paths_total[i]).replace('.tif', '.xml')

        with openslide.open_slide(str(truth_slide_path)) as truth:


          slide = openslide.open_slide(slide_paths[i])
          num_rows = int((bboxt[1] - bboxt[0]) / 256)
          num_columns = int((bboxt[3] - bboxt[2]) / 256)

          annotations = convert_xml_df(str(Anno_pathxml))
          x_values = list(annotations['X'].get_values())
          y_values = list(annotations['Y'].get_values())
          bbox = math.floor(np.min(x_values)), math.floor(np.max(x_values)), math.floor(np.min(y_values)), math.floor(np.max(y_values))
          m=0
          # height = math.floor(np.min(x_values)) - math.floor(np.max(x_values))
          # width = math.floor(np.min(y_values)) - math.floor(np.max(y_values))
          # print(height)
          # print(width)
          #a = []
          tile_index = 0
          for j in range(num_columns):
              for k in range(num_rows):
                r = random_crop2
                # mask = r[2].reshape(256, 256)
                print(crop_size[0])
                print(cv2.countNonZero(r[2]))
                tile_index = tile_index +1
                if (cv2.countNonZero(r[2]) > 1):
                 # print(r[2])
                 imageio.imwrite('E:\\camelyon16\\level1/tumor/%s_%d_%d.png' % (osp.splitext(osp.basename(slide_paths_total[i]))[0], r[3][1], r[3][0]), r[0])

                 imageio.imwrite('E:\\camelyon16\\level1\\mask/%s_%d_%d_mask.png' % (osp.splitext(osp.basename(slide_paths_total[i]))[0], r[3][1], r[3][0]), r[2])
    # else:
    #       while n in range(0, 80):
    #         slide = openslide.open_slide(slide_paths[i])
    #         r=random_crop_normal(slide, thresh, crop_size, bboxt)
    #
    #         if (cv2.countNonZero(r[1]) > crop_size[0]*crop_size[1]*0.2):
    #           print(r[0],'r0')
    #           print(r[2],'r2')
    #           print(r[1],'r1')
    #           imageio.imwrite('E:/camelyon16/level2/2_centers/normal/%s_%d_%d.png' % (osp.splitext(osp.basename(slide_paths[i]))[0], r[3][0], r[3][1]),r[0])
    #           imageio.imwrite('E:/camelyon16/level2/2_centers/nmask/%s_%d_%d.png' % (osp.splitext(osp.basename(slide_paths[i]))[0], r[3][0], r[3][1]),r[2])
    #
    #
    #           #b.append(r[3])
    #           #zz = testduplicates(b)
    #
    #           n=n+1
    #
    #         else:
    #          n=n
    #
    # else:
    #     o=0
    #     slide = openslide.open_slide(slide_paths_total[i])
    #     #c=[]
    #     while o in range(0, 500):
    #         nr=random_crop_normal(slide, thresh, crop_size, bboxt)
    #         if (cv2.countNonZero(r[1]) > crop_size[0]*crop_size[1]*0.2) and (o <= 1000):
    #            nmask = np.zeros((256, 256))
    #
    #            imageio.imwrite('E:/camelyon16/test/patches/normal/%s_%d_%d.png' % (osp.splitext(osp.basename(slide_paths_total[i]))[0], nr[2][0], nr[2][1]),nr[0])
    #            imageio.imwrite('E:/camelyon16/test/patches/mask_normal/%s_%d_%d_mask.png' % (osp.splitext(osp.basename(slide_paths_total[i]))[0], nr[2][0], nr[2][1]), nmask)
    #
    #            #c.append(r[3])
    #
    #            #zzz = testduplicates(c)
    #            o=o+1
    #
    #         else:
    #          o=o

    i=i+1
