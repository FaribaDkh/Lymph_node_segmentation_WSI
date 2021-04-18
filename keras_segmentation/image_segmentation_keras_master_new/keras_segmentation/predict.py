import glob
import random
import json
import os
import six

import cv2
import numpy as np
from tqdm import tqdm
from time import time
from PIL import Image
import os.path as osp
import matplotlib.pyplot as plt

from keras_segmentation.image_segmentation_keras_master_new.keras_segmentation.train import find_latest_checkpoint
from keras_segmentation.image_segmentation_keras_master_new.keras_segmentation.data_utils.data_loader import get_image_array, get_segmentation_array,\
    DATA_LOADER_SEED, class_colors, get_pairs_from_paths
from .models.config import IMAGE_ORDERING


random.seed(DATA_LOADER_SEED)


def model_from_checkpoint_path(checkpoints_path):

    from keras_segmentation.image_segmentation_keras_master_new.keras_segmentation.models.all_models import model_from_name
    assert (os.path.isfile(checkpoints_path+"_config.json")
            ), "Checkpoint not found."
    model_config = json.loads(open(checkpoints_path+ "_config.json ", "r").read())
    # model_config = json.loads(
    #     open(checkpoints_path+"_config.json", "r").read())
    # latest_weights = find_latest_checkpoint(checkpoints_path)
    # latest_weights = checkpoints_path+"model_unet_sgd.h5"

    latest_weights = checkpoints_path + "model.h5"
    assert (latest_weights is not None), "Checkpoint not found."
    model = model_from_name[model_config['model_class']](
        model_config['n_classes'], input_height=model_config['input_height'],
        input_width=model_config['input_width'])
    print("loaded weights ", latest_weights)
    model.load_weights(latest_weights,by_name = True)
    return model


def get_colored_segmentation_image(seg_arr, n_classes, colors=class_colors):
    output_height = seg_arr.shape[0]
    output_width = seg_arr.shape[1]
    seg_img = np.zeros((output_height, output_width, 3))

    for c in range(n_classes):
        seg_arr_c = seg_arr[:, :] == c
        seg_img[:, :, 0] += ((seg_arr_c)*(colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((seg_arr_c)*(colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((seg_arr_c)*(colors[c][2])).astype('uint8')

    return seg_img


def get_legends(class_names, colors=class_colors):

    n_classes = len(class_names)
    legend = np.zeros(((len(class_names) * 25) + 25, 125, 3),
                      dtype="uint8") + 255

    class_names_colors = enumerate(zip(class_names[:n_classes],
                                       colors[:n_classes]))

    for (i, (class_name, color)) in class_names_colors:
        color = [int(c) for c in color]
        cv2.putText(legend, class_name, (5, (i * 25) + 17),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
        cv2.rectangle(legend, (100, (i * 25)), (125, (i * 25) + 25),
                      tuple(color), -1)

    return legend


def overlay_seg_image(inp_img, seg_img):
    orininal_h = inp_img.shape[0]
    orininal_w = inp_img.shape[1]
    seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))

    fused_img = (inp_img/2 + seg_img/2).astype('uint8')
    return fused_img


def concat_lenends(seg_img, legend_img):

    new_h = np.maximum(seg_img.shape[0], legend_img.shape[0])
    new_w = seg_img.shape[1] + legend_img.shape[1]

    out_img = np.zeros((new_h, new_w, 3)).astype('uint8') + legend_img[0, 0, 0]

    out_img[:legend_img.shape[0], :  legend_img.shape[1]] = np.copy(legend_img)
    out_img[:seg_img.shape[0], legend_img.shape[1]:] = np.copy(seg_img)

    return out_img


def visualize_segmentation(seg_arr, inp_img=None, n_classes=None,
                           colors=class_colors, class_names=None,
                           overlay_img=False, show_legends=False,
                           prediction_width=None, prediction_height=None):

    if n_classes is None:
        n_classes = np.max(seg_arr)

    seg_img = get_colored_segmentation_image(seg_arr, n_classes, colors=colors)

    if inp_img is not None:
        orininal_h = inp_img.shape[0]
        orininal_w = inp_img.shape[1]
        seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))

    if (prediction_height is not None) and (prediction_width is not None):
        seg_img = cv2.resize(seg_img, (prediction_width, prediction_height))
        if inp_img is not None:
            inp_img = cv2.resize(inp_img,
                                 (prediction_width, prediction_height))

    if overlay_img:
        assert inp_img is not None
        seg_img = overlay_seg_image(inp_img, seg_img)

    if show_legends:
        assert class_names is not None
        legend_img = get_legends(class_names, colors=colors)

        seg_img = concat_lenends(seg_img, legend_img)

    return seg_img


def predict(model=None, inp=None, out_fname=None,
            checkpoints_path=None, overlay_img=False,
            class_names=None, show_legends=False, colors=class_colors,
            prediction_width=None, prediction_height=None, annotations_dir=None, output_dir=None, visualize=None):

    if model is None and (checkpoints_path is not None):
        model = model_from_checkpoint_path(checkpoints_path)

    assert (inp is not None)
    assert ((type(inp) is np.ndarray) or isinstance(inp, six.string_types)),\
        "Input should be the CV image or the input file name"

    if isinstance(inp, six.string_types):
        inp = cv2.imread(inp)
    assert len(inp.shape) == 3, "Image should be h,w,3 "

    output_width = model.output_width
    output_height = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes

    x = get_image_array(inp, input_width, input_height,
                        ordering=IMAGE_ORDERING)

    pr = model.predict(np.array([x]), verbose=1)

    pr = pr.reshape((output_height,  output_width, n_classes)).argmax(axis=2)
    seg_img = visualize_segmentation(pr, inp, n_classes=n_classes,
                                     colors=colors, overlay_img=overlay_img,
                                     show_legends=show_legends,
                                     class_names=class_names,
                                     prediction_width=prediction_width,
                                     prediction_height=prediction_height)
    # print(out_fname)
    # annotations_dir = "E:\\camelyon16\\dataset_for_training\\level_2\\new_dataset\\test_mask1\\"
    # pr = (pr > 0.5).astype(np.uint8)
    import pandas as pd
    df_all = pd.DataFrame()

    filename = osp.splitext(osp.basename(out_fname))[0]
    # print(filename)
    if (visualize == True):
        mask = cv2.imread(annotations_dir + filename + ".png",1)
        mask = mask * 255
        visualize_all_results(inp, pr, seg_img, mask, filename, output_dir=output_dir)

        if out_fname is not None:
            path = output_dir + "/visualize"

            try:
                os.mkdir(path)
            except OSError:
                print("Creation of the directory %s failed" % path)
            else:
                print("Successfully created the directory %s " % path)
            cv2.imwrite(path+ "/" + filename + '.png', seg_img)


    return pr


def predict_multiple(model=None, inps=None, inp_dir=None, output_dir=None,
                     checkpoints_path=None, overlay_img=False,
                     class_names=None, show_legends=False, colors=class_colors,
                     prediction_width=None, prediction_height=None, annotations_dir = None, visualize = None):

    if model is None and (checkpoints_path is not None):
        model = model_from_checkpoint_path(checkpoints_path)

    if inps is None and (inp_dir is not None):
        inps = glob.glob(os.path.join(inp_dir, "*.jpg")) + glob.glob(
            os.path.join(inp_dir, "*.png")) + \
            glob.glob(os.path.join(inp_dir, "*.jpeg"))


    assert type(inps) is list

    all_prs = []

    for i, inp in enumerate(tqdm(inps)):
        if output_dir is None:
            out_fname = None
        else:
            if isinstance(inp, six.string_types):
                out_fname = os.path.join(output_dir, os.path.basename(inp))
            else:
                out_fname = os.path.join(output_dir, str(i) + ".png")
                glob.glob(os.path.join(inp_dir, "*.jpg")) + glob.glob(
                    os.path.join(inp_dir, "*.png")) + \
                glob.glob(os.path.join(inp_dir, "*.jpeg"))


        pr = predict(model, inp, out_fname,
                     overlay_img=overlay_img, class_names=class_names,
                     show_legends=show_legends, colors=colors,
                     prediction_width=prediction_width,
                     prediction_height=prediction_height,annotations_dir = annotations_dir,output_dir= output_dir,visualize= visualize)

        all_prs.append(pr)

    return all_prs

def visualize_all_results(inp,pr,seg_img, mask,filename,output_dir= None):
    import os

    # define the name of the directory to be created

    # path = output_dir+"/output"
    path = output_dir

    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)

    seg_img.astype(np.int32)
    # inp.astype(np.int32)
    dst = cv2.addWeighted(seg_img, 0.5, inp[:,:,::-1], 0.5, 0.0, dtype=cv2.CV_32F)
    alpha = 0.2
    blended = alpha * inp + (1 - alpha) * seg_img

    pre = pr.astype(np.uint8)
    fig, ax = plt.subplots(1, 4, figsize=(10, 4))
    inp = inp.astype('uint8')
    ax[0].imshow(inp[:,:,::-1])
    seg_img.astype('uint8')
    # ax[1].imshow(predict[:, :, 1], cmap=plt.cm.gray)
    cmap = plt.cm.jet
    cmap.set_bad('w', 1.)

    ax[1].imshow(pre.squeeze(), vmin=0, vmax=1)
    # ax[3].imshow(np.argmax(predict, axis=2), cmap=plt.cm.gray)
    ax[2].imshow(mask)
    dst = dst.astype(np.uint8)
    ax[3].imshow(dst)
    # ax[4].imshow(seg_img)
    # cv2.imwrite(dst_path +"dst.png",dst)
    ax[0].title.set_text('Original Image')
    ax[1].title.set_text('Predicted Output')
    ax[2].title.set_text('GroundTruth')
    ax[3].title.set_text('Overlay Result')
    # ax[4].title.set_text('Overlaid output')
    plt.savefig(path + "/" + "result_" + filename + ".png", bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    plt.clf()
    plt.cla()


def evaluate(model=None, inp_images=None, annotations=None,
             inp_images_dir=None, annotations_dir=None, checkpoints_path=None, output_dir = None):
    text_file = open(output_dir + "Output_all.txt", "w")

    if model is None:
        assert (checkpoints_path is not None),\
                "Please provide the model or the checkpoints_path"
        model = model_from_checkpoint_path(checkpoints_path)

    if inp_images is None:
        assert (inp_images_dir is not None),\
                "Please provide inp_images or inp_images_dir"
        assert (annotations_dir is not None),\
            "Please provide inp_images or inp_images_dir"

        paths = get_pairs_from_paths(inp_images_dir, annotations_dir)
        paths = list(zip(*paths))
        inp_images = list(paths[0])
        annotations = list(paths[1])

    assert type(inp_images) is list
    assert type(annotations) is list

    tp = np.zeros(model.n_classes)
    fp = np.zeros(model.n_classes)
    fn = np.zeros(model.n_classes)
    tn = np.zeros(model.n_classes)

    tp_each_patch = np.zeros(model.n_classes)
    fp_each_patch = np.zeros(model.n_classes)
    fn_each_patch = np.zeros(model.n_classes)
    tn_each_patch = np.zeros(model.n_classes)

    n_pixels = np.zeros(model.n_classes)
    dice_coefficient_one_by_one = np.zeros(model.n_classes)
    f1_one_by_one = np.zeros(model.n_classes)
    jacard_one_by_one = np.zeros(model.n_classes)
    for inp, ann in tqdm(zip(inp_images, annotations)):
        out_fname = os.path.join(inp_images_dir, os.path.basename(inp))
        pr = predict(model, inp,out_fname=out_fname,annotations_dir=annotations_dir,output_dir = output_dir)

        gt = get_segmentation_array(ann, model.n_classes,
                                    model.output_width, model.output_height,
                                    no_reshape=True)
        gt = gt.argmax(-1)
        pr = pr.flatten()
        gt = gt.flatten()
        filename = osp.splitext(osp.basename(out_fname))[0]
        for cl_i in range(model.n_classes):
            tp[cl_i] += np.sum((pr == cl_i) * (gt == cl_i))
            fp[cl_i] += np.sum((pr == cl_i) * ((gt != cl_i)))
            fn[cl_i] += np.sum((pr != cl_i) * ((gt == cl_i)))
            tn[cl_i] += np.sum((pr != cl_i) * ((gt != cl_i)))
            n_pixels[cl_i] += np.sum(gt == cl_i)
            tp_each_patch[cl_i] = np.sum((pr == cl_i) * (gt == cl_i))
            fp_each_patch[cl_i] = np.sum((pr == cl_i) * ((gt != cl_i)))
            fn_each_patch[cl_i] = np.sum((pr != cl_i) * ((gt == cl_i)))
            tn_each_patch[cl_i] = np.sum((pr != cl_i) * ((gt != cl_i)))
        dice_coefficient_each_path = 2 * tp_each_patch / (fn_each_patch + (2 * tp_each_patch) + fp_each_patch)
        jacard_index_each_patch = tp_each_patch / (fn_each_patch + tp_each_patch + fp_each_patch)
        precision_each_patch = tp_each_patch / (tp_each_patch + fp_each_patch)
        recall_each_patch = tp_each_patch / (tp_each_patch + fn_each_patch)
        f1_each_patch = 2 * (precision_each_patch * recall_each_patch) / (precision_each_patch + recall_each_patch)
        dice_coefficient_one_by_one += dice_coefficient_each_path
        jacard_one_by_one += jacard_index_each_patch
        f1_one_by_one += f1_each_patch

    dice = dice_coefficient_one_by_one/(len(inp_images_dir))
    f1_one_by_one_all = f1_one_by_one/(len(inp_images_dir))
    jacard = jacard_one_by_one/(len(inp_images_dir))
    dice_coefficient = 2 * tp / (fn + (2*tp) + fp)
    jacard_index = tp / (fn + tp + fp)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2* (precision*recall)/(precision+recall)
    cl_wise_score = tp / (tp + fp + fn + 0.000000000001)
    n_pixels_norm = n_pixels / np.sum(n_pixels)
    frequency_weighted_IU = np.sum(cl_wise_score*n_pixels_norm)
    mean_IU = np.mean(cl_wise_score)
    text_file.write("Dice coefficient all : %s" % dice_coefficient)
    text_file.write("\n")
    text_file.write("f1 all : %s" % f1)
    text_file.write("\n")
    text_file.write("Jacard index : %s" % jacard_index)
    text_file.write("\n")
    text_file.write("________________________________")
    text_file.close()
    print("____________")
    print("frequency_weighted_IU:", frequency_weighted_IU)
    print("mean_IU: ", mean_IU)
    print("dice_coefficient: ", dice_coefficient)
    print("jacard_index: ", jacard_index)
    print("f1: ", f1)
    return {
        "frequency_weighted_IU": frequency_weighted_IU,
        "mean_IU": mean_IU,
        "class_wise_IU": cl_wise_score,
        "dice_coefficient: ": dice_coefficient,
        "jacard_index: ": jacard_index,
        "f1: ": f1
    }

def evaluate_binary(model=None, inp_images=None, annotations=None,
             inp_images_dir=None, annotations_dir=None, checkpoints_path=None, output_dir = None, source_path = None,visualize = None):
    import pandas as pd
    # filepath_name = os.path.basename(os.path.split(checkpoints_path))

    filenameMask, e = os.path.splitext(checkpoints_path)
    filenameMask1, e = os.path.splitext(checkpoints_path)

    # a = filenameMask.split("\\", 2)[1:]
    b = filenameMask1.rsplit('\\', 3)
    df_sns_data_split_10k = pd.DataFrame({"Type": [], "TP": [], "TN": [], "FP": [], "FN": [],"File_name":[]})
    # text_file = open(output_dir + filepath_name +"_Output.txt", "w")
    # checkpoint_name = b[1]+"_"+b[2]
    if model is None:
        assert (checkpoints_path is not None),\
                "Please provide the model or the checkpoints_path"
        model = model_from_checkpoint_path(checkpoints_path)

    if inp_images is None:
        assert (inp_images_dir is not None),\
                "Please provide inp_images or inp_images_dir"
        assert (annotations_dir is not None),\
            "Please provide inp_images or inp_images_dir"

        paths = get_pairs_from_paths(inp_images_dir, annotations_dir)
        paths = list(zip(*paths))
        inp_images = list(paths[0])
        annotations = list(paths[1])

    assert type(inp_images) is list
    assert type(annotations) is list

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    tp_each_patch = 0
    fp_each_patch = 0
    fn_each_patch = 0
    tn_each_patch = 0

    n_pixels = 0
    dice_coefficient_one_by_one = 0
    f1_one_by_one = 0
    jacard_one_by_one = 0
    num_images = len(inp_images)
    for inp, ann in tqdm(zip(inp_images, annotations)):
        out_fname = os.path.join(inp_images_dir, os.path.basename(inp))
        pr = predict(model, inp,out_fname=out_fname,annotations_dir=annotations_dir,output_dir = output_dir, visualize = None)
        input_name, e = os.path.splitext(inp)
        print(input_name)
        gt = get_segmentation_array(ann, model.n_classes,
                                    model.output_width, model.output_height,
                                    no_reshape=True)
        gt = gt.argmax(-1)
        pr = pr.flatten()
        gt = gt.flatten()
        filename = osp.splitext(osp.basename(out_fname))[0]
        confusion_matrix_arrs = {}

        gt_inverse = np.logical_not(gt)
        pr_inverse = np.logical_not(pr)

        confusion_matrix_arrs['tp'] = np.logical_and(gt, pr)
        tp_each_patch = np.count_nonzero(confusion_matrix_arrs['tp'])
        confusion_matrix_arrs['tn'] = np.logical_and(gt_inverse, pr_inverse)
        tn_each_patch = np.count_nonzero(confusion_matrix_arrs['tn'])
        confusion_matrix_arrs['fp'] = np.logical_and(gt_inverse, pr)
        fp_each_patch = np.count_nonzero(confusion_matrix_arrs['fp'])
        confusion_matrix_arrs['fn'] = np.logical_and(gt, pr_inverse)
        fn_each_patch = np.count_nonzero(confusion_matrix_arrs['fn'])
        df_sns_data_split_10k = df_sns_data_split_10k.append({"Type": "Five_layers_without_BN",
                                                              "TP": tp_each_patch,
                                                              "TN": tn_each_patch,
                                                              "FP": fp_each_patch,
                                                              "FN": fn_each_patch,"File_name":filename},
                                                             ignore_index=True)
        tp += tp_each_patch
        tn += tn_each_patch
        fp += fp_each_patch
        fn += fn_each_patch

    # df_sns_data_split_10k.to_csv(source_path +filenameMask +'center2.csv')
    df_sns_data_split_10k.to_csv(source_path +'wbn.csv')

    dice_coefficient = 2 * tp / (fn + (2*tp) + fp + 0.000000000001)
    jacard_index = tp / (fn + tp + fp + 0.000000000001)
    precision = tp/(tp+fp+ 0.000000000001)
    recall = tp/(tp+fn+ 0.000000000001)
    f1 = 2 * (precision*recall)/(precision+recall+ 0.000000000001)
    cl_wise_score = tp / (tp + fp + fn + 0.000000000001)
    return (tp, tn, fp, fn, dice_coefficient,f1,jacard_index)

