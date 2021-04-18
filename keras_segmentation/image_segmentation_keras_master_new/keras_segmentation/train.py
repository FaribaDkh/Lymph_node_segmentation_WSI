import json
# from keras.preprocessing.image import ImageDataGenerator
import tensorflow.compat.v1 as tf

from keras_segmentation.image_segmentation_keras_master_new.keras_segmentation.data_utils.data_loader import image_segmentation_generator, \
    verify_segmentation_dataset
import glob
import six
import os


import tensorflow as tf
from keras.callbacks import Callback, ModelCheckpoint
import matplotlib.pyplot as plt
from keras.optimizers import *
# import wandb
# wandb.init()
from tensorflow import keras
from keras.callbacks import EarlyStopping
import pandas as pd

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)


def find_latest_checkpoint(checkpoints_path, fail_safe=True):

    def get_epoch_number_from_path(path):
        return path.replace(checkpoints_path, "").strip(".")

    # Get all matching files
    all_checkpoint_files = glob.glob(checkpoints_path+".*")
    # Filter out entries where the epoc_number part is pure number
    all_checkpoint_files = list(filter(lambda f: get_epoch_number_from_path(f)
                                       .isdigit(), all_checkpoint_files))
    if not len(all_checkpoint_files):
        # The glob list is empty, don't have a checkpoints_path
        if not fail_safe:
            raise ValueError("Checkpoint path {0} invalid"
                             .format(checkpoints_path))
        else:
            return None

    # Find the checkpoint file with the maximum epoch
    latest_epoch_checkpoint = max(all_checkpoint_files,
                                  key=lambda f:
                                  int(get_epoch_number_from_path(f)))
    return latest_epoch_checkpoint


def masked_categorical_crossentropy(gt, pr):
    from keras.losses import categorical_crossentropy
    mask = 1 - gt[:, :, 0]
    return categorical_crossentropy(gt, pr) * mask

def my_loss_fn(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)

def dice_loss(gt, pr):
  numerator = 2 * tf.reduce_sum(gt * pr, axis=(1,2,3))
  denominator = tf.reduce_sum(gt + pr, axis=(1,2,3))

  return 1 - numerator / denominator


# Define IoU metric
import numpy as np
from keras import backend as K
def mean_iou(y_true, y_pred):
    prec = []
    from tensorflow.python.keras.metrics import Metric
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.cast(y_pred > t, tf.int32)
        score, up_opt = tf.compat.v1.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

class CheckpointsCallback(Callback):
    def __init__(self, checkpoints_path):
        self.checkpoints_path = checkpoints_path

    def on_epoch_end(self, epoch, logs=None):
        if self.checkpoints_path is not None:
            self.model.save_weights(self.checkpoints_path + str(epoch))
            # print("saved ", self.checkpoints_path + str(epoch))


def train(model,
          train_images,
          train_annotations,
          input_height=None,
          input_width=None,
          n_classes=None,
          verify_dataset=False,
          checkpoints_path=None,
          epochs=5,
          batch_size=2,
          validate=True,
          val_images=None,
          val_annotations=None,
          val_batch_size=2,
          auto_resume_checkpoint=False,
          load_weights=None,
          steps_per_epoch=512,
          val_steps_per_epoch=512,
          gen_use_multiprocessing=False,
          ignore_zero_class=False,
          optimizer_name='adadelta',
          do_augment=False,
          augmentation_name="aug_all"):

    from keras_segmentation.image_segmentation_keras_master_new.keras_segmentation.models.all_models import model_from_name
    # opt = tf.keras.optimizers.SGD(learning_rate=0.001)
    # check if user gives model name instead of the model object
    if isinstance(model, six.string_types):
        # create the model from the name
        assert (n_classes is not None), "Please provide the n_classes"
        if (input_height is not None) and (input_width is not None):
            model = model_from_name[model](
                n_classes, input_height=input_height, input_width=input_width)
        else:
            model = model_from_name[model](n_classes)

    n_classes = model.n_classes
    input_height = model.input_height
    input_width = model.input_width
    output_height = model.output_height
    output_width = model.output_width

    if validate:
        assert val_images is not None
        assert val_annotations is not None

        opt = Adam(lr=1E-6, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        # opt = tf.keras.optimizers.SGD(learning_rate=0.001)
        model.compile(optimizer=opt, loss="binary_crossentropy", metrics=['accuracy'])

    if checkpoints_path is not None:
        with open(checkpoints_path+"_config.json", "w") as f:
            json.dump({
                "model_class": model.model_name,
                "n_classes": n_classes,
                "input_height": input_height,
                "input_width": input_width,
                "output_height": output_height,
                "output_width": output_width
            }, f)

    if load_weights is not None and len(load_weights) > 0:
        print("Loading weights from ", load_weights)
        model.load_weights(load_weights)

    if auto_resume_checkpoint and (checkpoints_path is not None):
        latest_checkpoint = find_latest_checkpoint(checkpoints_path)
        if latest_checkpoint is not None:
            print("Loading the weights from latest checkpoint ",
                  latest_checkpoint)
            model.load_weights(latest_checkpoint)

    if verify_dataset:
        print("Verifying training dataset")
        verified = verify_segmentation_dataset(train_images,
                                               train_annotations,
                                               n_classes)
        assert verified
        if validate:
            print("Verifying validation dataset")
            verified = verify_segmentation_dataset(val_images,
                                                   val_annotations,
                                                   n_classes)
            assert verified

    train_gen = image_segmentation_generator(
        train_images, train_annotations,  batch_size,  n_classes,
        input_height, input_width, output_height, output_width,
        do_augment=do_augment, augmentation_name=augmentation_name)

    if validate:
        val_gen = image_segmentation_generator(
            val_images, val_annotations,  val_batch_size,
            n_classes, input_height, input_width, output_height, output_width)
    # simple early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=5)
    # callbacks = [
    #     CheckpointsCallback(checkpoints_path),es
    # ]
    callbacks = [EarlyStopping(monitor='val_loss', patience=20),
                 ModelCheckpoint(filepath='model.h5', monitor='val_loss', save_best_only=True)]
    # callbacks = [es]
    if not validate:
        model.fit_generator(train_gen, steps_per_epoch,
                            epochs=epochs, callbacks=callbacks)
    else:
        history = model.fit_generator(train_gen,steps_per_epoch,validation_data=val_gen,
                            validation_steps=val_steps_per_epoch,
                            epochs=epochs, callbacks=callbacks,
                            use_multiprocessing=gen_use_multiprocessing)


    # model.save_weights(checkpoints_path + "model.h5", save_format='h5')
    model.save_weights(checkpoints_path + "model.h5")

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(checkpoints_path + 'accuracy.png', dpi=100)
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(checkpoints_path + 'loss.png', dpi=100)
    plt.show()


n_classes = 2
input_height = 256
input_width = 256
dice_value = []
# train paths
"E:\\camelyon16\\new_dataset\\splitted_dataset\\"
train_images = "E:\\camelyon16\\new_dataset\\splitted_dataset\\train\\"
train_annotations = "E:\\camelyon16\\new_dataset\\splitted_dataset\\train_mask_1\\"
val_images = "E:\\camelyon16\\new_dataset\\splitted_dataset\\val\\"
val_annotations = "E:\\camelyon16\\new_dataset\\splitted_dataset\\val_mask_1\\"
if __name__ == "__main__":
    from keras_segmentation.models.unet import unet,unet_mini,unet_three_layer
    from keras_segmentation.models.fcn import fcn_8
    # from keras_segmentation import train
    from keras_segmentation.predict import predict, evaluate, predict_multiple, evaluate_binary
    import time
    import pandas as pd
    start_time = time.time()
    import seaborn as sns
    import seaborn as sns


    # all_checkpoints = "E:\\image_segmentation_keras_master_new\\checkpoints\\50k\\reinhard_center2\\"
    all_checkpoints = "E:\\image_segmentation_keras_master_new\\checkpoints\\kmeans\\wbn_mini\\"
    # model = unet(n_classes=2, input_height=256, input_width=256,encoder_level= 3)
    model = unet_mini(n_classes=2, input_height=256, input_width=256)
    # # # # model = unet_three_layer(n_classes=2, input_height=256, input_width=256)
    # # # model = unet(n_classes=2, input_height=256, input_width=256,encoder_level=5)
    # # # # # #
    train(model=model, train_images=train_images, train_annotations=train_annotations, input_height=256, input_width=256,
          val_images=val_images, val_annotations=val_annotations, n_classes=2, batch_size=16
          , val_batch_size=16, epochs=100, checkpoints_path=all_checkpoints, do_augment=False, optimizer_name='SGD')
    print("--- %s min ---" % float((time.time() - start_time)/60))


    # ###### test
    df_sns = pd.DataFrame({"slide_name": [], "level": [], "image_type": [], "dice_coeff": [], "jacard_index": [], "F1":[]})
    df_sns_data_split_color_normalization = pd.DataFrame({"type":[],"Level":[],"dice_coeff":[],"jacard_index":[], "F1":[]})
    df_sns_data_split_10k = pd.DataFrame({"type": [], "TP": [],"TN": [],"FP": [],"FN":[], "dice_coeff": [],"jacard_index": [], "F1": []})
    # #
    source_path = "E:\\camelyon16\\new_dataset\\splitted_dataset\\"
    predict_multiple(inp_dir=source_path + "\\test\\",
                     output_dir=source_path+"\\results\\wbn_mini\\",
                     annotations_dir=source_path + "\\test_mask\\",
                     checkpoints_path=all_checkpoints,
                     overlay_img=True, visualize=True)

    results_original_org = evaluate_binary(inp_images_dir=source_path + "\\test\\",\
                                         annotations_dir=source_path + "\\test_mask\\",\
                                         checkpoints_path= all_checkpoints,
                                         output_dir=source_path + "\\results\\wbn_mini\\",\
                                         visualize=False,source_path=source_path)


