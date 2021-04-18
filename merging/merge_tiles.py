import sys
import os
import numpy as np
from PIL import Image
import cv2
import os, sys
# path = "E:\\camelyon16\\salar\\"

# path = "E:\\camelyon16\\TrainingData\\merge_tile\\14_level_0\\imgs\\"
output = "E:\\camelyon16\\camelyon2017\\"
patchdir = os.listdir(output)
patch_size = 256
# images = [Image.open(x) for x in ['Test1.jpg', 'Test2.jpg', 'Test3.jpg']]
# widths, heights = zip(*(i.size for i in images))
# num_columns = 47
# num_rows= 118

# #
# num_columns = 383
# num_rows= 837
# num_columns = 73
# num_rows= 55

#
# total_width = sum(widths)
# max_height = max(heights)
path = "E:\\camelyon16\\camelyon2017\\"
dirs = os.listdir(path)
def all_slides():
    for i in dirs:
        filename, e = os.path.splitext(i)
        f = open(path + filename+"\\"+ filename+".txt", "r")
        text = f.readlines()
        columns = int(text[2].split(": ")[1])
        rows = int(text[3].split(": ")[1])
        merge_tile(columns,rows,filename)

def merge_tile(num_columns, num_rows, filename):
    total_width = num_columns * patch_size
    max_height = num_rows * patch_size
    #
    new_im = Image.new('RGB', (total_width, max_height))
    new_im_mask = Image.new('L', (total_width, max_height))

    x_offset = 0
    y_offset = 0
    i = 0
    j = 0
    image_size = 0
    for i in range(num_rows):
        for j in range(num_columns):
            if os.path.isfile(path + filename + "\\patches\\" + str(image_size) + ".png"):
                # im = Image.open(path + filename + "\\patches\\"+  filename +"_"+ str(image_size) + ".png")
                im = Image.open(path + filename + "\\results\\"+ str(image_size) + ".png")
                # im.save('test1.png')
            else:
                im = Image.new('RGB', (patch_size, patch_size))
                # im_mask = Image.new('L', (patch_size, patch_size))
                # im.save('test1.jpg')
            image_size = image_size + 1

            new_im.paste(im, (x_offset, y_offset))
            # new_im_mask.paste(im_mask, (x_offset, y_offset))
            x_offset += im.size[0]
            # x_offset += im_mask.size[0]
        x_offset = 0
        y_offset += im.size[0]
        # y_offset += im_mask.size[0]
    rescaling = pow(2, 10)
    new_im.thumbnail((int(total_width / 8), int(total_width / 8)), Image.ANTIALIAS)
    # new_im_mask.thumbnail((int(total_width / 12), int(total_width / 12)), Image.ANTIALIAS)
    # new_im.resize((int(total_width/rescaling),int(total_width/rescaling)), resample=0)
    new_im.save(output+ filename +"/" +filename + "_merged_mask.png")
    # new_im_mask.save(output + filename + "_merged_mask.png")

if __name__ == '__main__':
    all_slides()
# new_im = np.array(new_im)
# # Convert RGB to BGR
# new_im = new_im[:, :, ::-1].copy()
# scale_percent = 1  # percent of original size
# width = int(new_im.shape[1] * scale_percent / 100)
# height = int(new_im.shape[0] * scale_percent / 100)
# dim = (width, height)
# # resize image
# resized = cv2.resize(new_im, dim, interpolation=cv2.INTER_AREA)

# print('Resized Dimensions : ', resized.shape)
#
# cv2.imshow("Resized image", resized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imwrite('grad_cam_14.png',resized)
# resized.save('test.jpg')
