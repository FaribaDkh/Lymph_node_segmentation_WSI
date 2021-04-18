from keras.models import *
from keras.layers import *

from .config import IMAGE_ORDERING
from .model_utils import get_segmentation_model
from .vgg16 import get_vgg_encoder
from .mobilenet import get_mobilenet_encoder
from .basic_models import vanilla_encoder
from .resnet50 import get_resnet50_encoder



if IMAGE_ORDERING == 'channels_first':
    MERGE_AXIS = 1
elif IMAGE_ORDERING == 'channels_last':
    MERGE_AXIS = -1
def unet_three_layer(n_classes, input_height=360, input_width=480):

    if IMAGE_ORDERING == 'channels_first':
        img_input = Input(shape=(3, input_height, input_width))
    elif IMAGE_ORDERING == 'channels_last':
        img_input = Input(shape=(input_height, input_width, 3))
    num_filters = 32
    kernel_size = (3, 3)
    conv1 = Conv2D(num_filters, kernel_size, activation='relu', padding='same')(img_input)
    # conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(num_filters, kernel_size, activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1)

    conv2 = Conv2D(num_filters * 2, kernel_size, activation='relu', padding='same')(pool1)
    # conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(num_filters * 2, kernel_size, activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2)

    conv3 = Conv2D(num_filters * 4, kernel_size, activation='relu', padding='same')(pool2)
    # conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(num_filters * 4, kernel_size, activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3)

    ### CENTER ###
    conv4 = Conv2D(num_filters * 8, kernel_size, activation='relu', padding='same')(pool3)
    # conv5 = BatchNormalization()(conv5)
    conv4 = Conv2D(num_filters * 8, kernel_size, activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)

    up6 = concatenate([Conv2DTranspose(num_filters * 4, (2, 2), strides=(2, 2), padding='same')(conv4), conv3],
                      axis=3)
    conv5 = Conv2D(num_filters * 4, kernel_size, activation='relu', padding='same')(up6)
    # conv7 = BatchNormalization()(conv7)
    conv5 = Conv2D(num_filters * 4, kernel_size, activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)

    up7 = concatenate([Conv2DTranspose(num_filters * 2, (2, 2), strides=(2, 2), padding='same')(conv5), conv2],
                      axis=3)
    conv6 = Conv2D(num_filters * 2, kernel_size, activation='relu', padding='same')(up7)
    # conv8 = BatchNormalization()(conv8)
    conv6 = Conv2D(num_filters * 2, kernel_size, activation='relu', padding='same')(conv6)
    # conv8 = BatchNormalization()(conv8)

    up8 = concatenate([Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(conv6), conv1],
                      axis=3)
    conv7 = Conv2D(num_filters, kernel_size, activation='relu', padding='same')(up8)
    # conv9 = BatchNormalization()(conv9)
    conv7 = Conv2D(num_filters, kernel_size, activation='relu', padding='same')(conv7)
    # conv9 = BatchNormalization()(conv9)

    o = Conv2D(n_classes, (1, 1), data_format=IMAGE_ORDERING,
               padding='same')(conv7)
    model = get_segmentation_model(img_input, o)
    model.model_name = "unet_three_layer"
    return model

def unet_mini(n_classes, input_height=360, input_width=480):

    if IMAGE_ORDERING == 'channels_first':
        img_input = Input(shape=(3, input_height, input_width))
    elif IMAGE_ORDERING == 'channels_last':
        img_input = Input(shape=(input_height, input_width, 3))
    num_filters = 32
    kernel_size = (3, 3)
    conv1 = Conv2D(num_filters, kernel_size, activation='relu', padding='same')(img_input)
    # conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(num_filters, kernel_size, activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1)

    conv2 = Conv2D(num_filters * 2, kernel_size, activation='relu', padding='same')(pool1)
    # conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(num_filters * 2, kernel_size, activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2)

    conv3 = Conv2D(num_filters * 4, kernel_size, activation='relu', padding='same')(pool2)
    # conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(num_filters * 4, kernel_size, activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3)

    conv4 = Conv2D(num_filters * 8, kernel_size, activation='relu', padding='same')(pool3)
    # conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(num_filters * 8, kernel_size, activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2))(conv4)

    ### CENTER ###
    conv5 = Conv2D(num_filters * 16, kernel_size, activation='relu', padding='same')(pool4)
    # conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(num_filters * 16, kernel_size, activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)

    ### DECODING PATH ###
    up6 = concatenate([Conv2DTranspose(num_filters * 8, (2, 2), strides=(2, 2), padding='same')(conv5), conv4],
                      axis=3)
    conv6 = Conv2D(num_filters * 8, kernel_size, activation='relu', padding='same')(up6)
    # conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(num_filters * 8, kernel_size, activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = concatenate([Conv2DTranspose(num_filters * 4, (2, 2), strides=(2, 2), padding='same')(conv6), conv3],
                      axis=3)
    conv7 = Conv2D(num_filters * 4, kernel_size, activation='relu', padding='same')(up7)
    # conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(num_filters * 4, kernel_size, activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = concatenate([Conv2DTranspose(num_filters * 2, (2, 2), strides=(2, 2), padding='same')(conv7), conv2],
                      axis=3)
    conv8 = Conv2D(num_filters * 2, kernel_size, activation='relu', padding='same')(up8)
    # conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(num_filters * 2, kernel_size, activation='relu', padding='same')(conv8)
    # conv8 = BatchNormalization()(conv8)

    up9 = concatenate([Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(conv8), conv1],
                      axis=3)
    conv9 = Conv2D(num_filters, kernel_size, activation='relu', padding='same')(up9)
    # conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(num_filters, kernel_size, activation='relu', padding='same')(conv9)
    # conv9 = BatchNormalization()(conv9)

    o = Conv2D(n_classes, (1, 1), data_format=IMAGE_ORDERING,
               padding='same')(conv9)
    model = get_segmentation_model(img_input, o)
    model.model_name = "unet_mini"
    return model


def _unet(n_classes, encoder, l1_skip_conn=True, input_height=416,
          input_width=608):

    img_input, levels = encoder(
        input_height=input_height, input_width=input_width)
    [f1, f2, f3, f4, f5] = levels

    o = f4
    o = f4

    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(512, (3, 3), padding='valid' , activation='relu' , data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, f3], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(256, (3, 3), padding='valid', activation='relu' , data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, f2], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(128, (3, 3), padding='valid' , activation='relu' , data_format=IMAGE_ORDERING))(o)
    # o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)

    if l1_skip_conn:
        o = (concatenate([o, f1], axis=MERGE_AXIS))

    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(64, (3, 3), padding='valid', activation='relu', data_format=IMAGE_ORDERING))(o)
    # o = (BatchNormalization())(o)

    o = Conv2D(n_classes, (3, 3), padding='same',
               data_format=IMAGE_ORDERING)(o)

    model = get_segmentation_model(img_input, o)

    return model


def unet(n_classes, input_height=416, input_width=608, encoder_level=3):

    model = _unet(n_classes, vanilla_encoder,
                  input_height=input_height, input_width=input_width)
    model.model_name = "unet"
    return model


def vgg_unet(n_classes, input_height=416, input_width=608, encoder_level=3):

    model = _unet(n_classes, get_vgg_encoder,
                  input_height=input_height, input_width=input_width)
    model.model_name = "vgg_unet"
    return model


def resnet50_unet(n_classes, input_height=416, input_width=608,
                  encoder_level=3):

    model = _unet(n_classes, get_resnet50_encoder,
                  input_height=input_height, input_width=input_width)
    model.model_name = "resnet50_unet"
    return model


def mobilenet_unet(n_classes, input_height=224, input_width=224,
                   encoder_level=3):

    model = _unet(n_classes, get_mobilenet_encoder,
                  input_height=input_height, input_width=input_width)
    model.model_name = "mobilenet_unet"
    return model


if __name__ == '__main__':
    m = unet_mini(101)
    m = _unet(101, vanilla_encoder)
    # m = _unet( 101 , get_mobilenet_encoder ,True , 224 , 224  )
    m = _unet(101, get_vgg_encoder)
    m = _unet(101, get_resnet50_encoder)
