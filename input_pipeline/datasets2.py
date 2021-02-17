import tensorflow as tf
import pandas as pd
import glob
import numpy as np
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split

'''User Defined Imports'''
import constants
from input_pipeline.preprocessing import resampling

AUTOTUNE = tf.data.experimental.AUTOTUNE


def _build_dataset(files, labels, data_set_type):
    """

    Args:
        files:
        labels:
        data_set_type:

    Returns:

    """
    # Create tf data set
    ds = tf.data.Dataset.from_tensor_slices((files, labels))

    if data_set_type == 'train':
        p_var = data_set_type
        print("Buildling {} data set".format(p_var))
        ds = ds.shuffle(constants.N_SHUFFLE_BUFFER)
        ds = ds.cache()
        ds = ds.map(_augment_parse, num_parallel_calls=AUTOTUNE)

    if data_set_type != 'train':
        print("Buildling {} data set".format(data_set_type))
        ds = ds.map(_parse_func, AUTOTUNE)

    ds = ds.batch(constants.N_BATCH_SIZE).prefetch(AUTOTUNE)
    print(ds.element_spec)
    return ds


@tf.function
def _augment_parse(a_filename, a_label):

    a_image_string = tf.io.read_file(a_filename)
    a_image_decoded = tf.io.decode_jpeg(a_image_string, channels=3)

    # original image dimension -2848*4288(H*W)
    # process image by reducing the black background
    a_image_bbcrp = tf.image.crop_to_bounding_box(a_image_decoded, 0, 266, 2848, 3426)

    a_image_normal = tf.cast(a_image_bbcrp, tf.float32) / 255.0

    a_image = tf.image.resize(a_image_normal, size=(256, 256))

    # a_image_crp1 = tf.image.central_crop(a_image_normal, 0.85)
    # augment by image flip and rotation
    a_image = tf.image.random_flip_left_right(a_image)
    a_image = tf.image.random_flip_up_down(a_image)
    rot_range = random.randint(24, 36)
    # below lone enables counterclockwise rotation and clockwise rotaion
    # rot_range = random.randrange(-36, 36, 1)
    a_image = tfa.image.rotate(a_image, tf.constant((np.pi / rot_range)),
                               interpolation='NEAREST')

    return a_image, a_label


@tf.function
def _parse_func(filename, label):

    image_string = tf.io.read_file(filename)
    image_decoded = tf.io.decode_jpeg(image_string, channels=3)
    # original image dimension -2848*4288(H*W)
    image_bbcrp = tf.image.crop_to_bounding_box(image_decoded, 0, 266, 2848, 3426)

    image_normal = tf.cast(image_bbcrp, tf.float32) / 255.0
    # image_crp1 = tf.image.central_crop(image_normal, 0.85)
    image = tf.image.resize(image_normal, size=(256, 256))

    # label = tf.one_hot(label) @ for multiclass classification
    return image, label


def load_data():
    """
    Returns: prefetched tensorflow dataset
    """
    tf_train_data, tf_valid_data, tf_test_data = _get_datasets()

    # print(np.shape(np_train_images), np.shape(np_train_labels))

    return tf_train_data, tf_valid_data, tf_test_data


def _get_datasets():
    """
    PURPOSE: Read raw data, reassign labels, resampling, building respective
             datasets of train, test,valid
    Returns: train data, test data, validation data

    """
    # list of image paths
    list_image_paths = glob.glob(constants.path_train_img + '/*')
    list_image_paths_test = glob.glob(constants.path_test_img + '/*')

    # List of labels
    df_imagenames_labels_train = pd.read_csv(constants.path_train_labels,
                                             usecols=constants.COLUMN_LABELS)
    df_imagenames_labels_test = pd.read_csv(constants.path_test_labels,
                                            usecols=constants.COLUMN_LABELS)

    # create a new column to store corresponding image paths
    df_imagenames_labels_train['img_paths'] = list_image_paths
    df_imagenames_labels_test['img_paths'] = list_image_paths_test

    # print(df_imagenames_labels_test.head())

    '''############################### Reassign labels for binary classification ##################################'''

    # process labels, categorize (0,1 = 0[NPR]), and (2,3,4 = 1[PR])
    df_imagenames_labels_train['Retinopathy grade'] = \
        df_imagenames_labels_train['Retinopathy grade'].map({0: 0, 1: 0, 2: 1, 3: 1, 4: 1})

    df_imagenames_labels_test['Retinopathy grade'] = \
        df_imagenames_labels_test['Retinopathy grade'].map({0: 0, 1: 0, 2: 1, 3: 1, 4: 1})
    print('Testing set:\n', df_imagenames_labels_test['Retinopathy grade'].value_counts())
    # print("check the image labels: \n", (df_imagenames_labels_train.head()))

    '''###### Random shuffle whole training data and split to train and validation #########'''

    df_train_unbal, df_valid = train_test_split(df_imagenames_labels_train, test_size=0.2, random_state=42)
    # print(df_train_unbal.head(-1), df_valid.head(-1))

    '''############################################################################################################'''

    '''  ######################################    resampling    #################################################  '''

    print('Before resampling:\n', df_train_unbal['Retinopathy grade'].value_counts())
    df_balanced = resampling(df_train_unbal, frac=1)
    print('After resampling:\n', df_balanced['Retinopathy grade'].value_counts())
    print("Shape of balanced train dataset:", df_balanced.shape)

    ''' ############################# Building train, valid ans test data set ################################### '''

    train_ds = _build_dataset(df_balanced['img_paths'].tolist(), df_balanced['Retinopathy grade'].astype(int).tolist(),
                              'train')

    valid_ds = _build_dataset(df_valid['img_paths'].tolist(), df_valid['Retinopathy grade'].astype(int).tolist(),
                              'valid')

    test_ds = _build_dataset(df_imagenames_labels_test['img_paths'].tolist(),
                             df_imagenames_labels_test['Retinopathy grade'].astype(int).tolist(), 'test')

    # plot samples in a grid from all sets
    _plot_images(train_ds, 'training samples')
    _plot_images(valid_ds, 'validation samples')
    _plot_images(test_ds, 'testing samples')

    ''' ############################################################################################################ '''

    return train_ds, valid_ds, test_ds


def _plot_images(dataset, dataset_name):
    """

    Args:
        dataset: dataset object from which images have to be plotted
        dataset_name: name of the type of split (train/test/valid)
    """
    plt.figure(figsize=(10, 10))
    plt.suptitle(dataset_name)
    for images, labels in dataset.take(1):
        labels_numpy = labels.numpy()
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy())
            # print((labels[i].numpy()))
            # plt.title("class:%d" % labels_numpy[i])
            plt.axis("on")
    plt.show()
    pass
