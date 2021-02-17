import gin
import logging
import tensorflow as tf
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from input_pipeline.preprocessing import preprocess, resampling
import constants
import glob
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.model_selection import train_test_split



# tf.compat.v1.enable_eager_execution()

print("Tensorflow version", tf.__version__)

AUTOTUNE = tf.data.experimental.AUTOTUNE


@gin.configurable
def load(name, data_dir):
    """

    Args:
        name: dataset name
        data_dir: dataset directory

    Returns: prefetched tensorflow datasets(train,validation,test)

    """
    if name == "idrid":
        logging.info(f"Preparing dataset {name}...")

        # ...
        # columns_from_labels = ['Image name', 'Retinopathy grade']
        columns_from_labels = constants.COLUMN_LABELS

        # get paths to all directories of images and labels.
        dir_train_images, dir_test_images, dir_train_csv, dir_test_csv = path2dir(data_dir)

        # loading csv files : pass directory paths to csv and parse columns, that are to be used to create a data frame
        df_names_labels_train = load_from_csv(dir_train_csv, columns_from_labels)
        df_names_labels_test = load_from_csv(dir_test_csv, columns_from_labels)

        print(df_names_labels_train['Retinopathy grade'].value_counts())

        print('There are %i train labels and %i test labels' % (len(df_names_labels_train), len(df_names_labels_test)))

        # Display a random image
        show_sample_image(dir_train_images)
        # attach file extensions to image names
        df_names_labels_train = _append_file_format_extension2name(df_names_labels_train)
        df_names_labels_test = _append_file_format_extension2name(df_names_labels_test)
        print(df_names_labels_test.head())

        '''##### Split train data into train and validation #####'''
        df_train, df_valid = train_test_split(df_names_labels_train, test_size=0.2, random_state=42)

        '''#### Over sample the TRaining dataset using, Resampling- using sample function of pandas dataframes #####'''
        df_resampled_data = resampling(df_train, frac=1)
        print(df_resampled_data['Retinopathy grade'].value_counts())
        '''###################################################################################'''

        ''' Training and validation data building '''
        gen_img_train_valid = ImageDataGenerator(preprocessing_function=preprocess,
                                                 rescale=1.0 / 255,
                                                 rotation_range=10,
                                                 horizontal_flip=False,
                                                 vertical_flip=True,
                                                 zoom_range=0.01)

        # Training data set build
        print("######################################################")
        print("Loading training Data ............")
        ds_train = _build_dataset(df_resampled_data, dir_train_images, gen_img_train_valid,
                                  class_mode='binary', subset_name=None, shuffle_val=True)
        print("# Finished Loading training Data #")
        print("######################################################")
        '''## No augmentation for validation and test data ##'''
        gen_img_valid = ImageDataGenerator(preprocessing_function=preprocess,
                                           rescale=1.0 / 255)
        # Validation data set build
        print("Loading Validation Data ............")
        ds_val = _build_dataset(df_valid, dir_train_images, gen_img_valid,
                                class_mode='binary', subset_name=None, shuffle_val=False)
        print("# Finished Loading Validation Data #")

        print("######################################################")
        ''' Testing dataset building  '''
        print("Loading Validation Data ............")

        gen_img_test = ImageDataGenerator(preprocessing_function=preprocess, rescale=1.0 / 255)

        ds_test = _build_dataset(df_names_labels_test, dir_test_images, gen_img_test,
                                 class_mode='binary', subset_name=None, shuffle_val=False)

        print("# Finished Loading Validation Data #")
        # Display a sample image along with label from training  data set
        _show_sample_from_ds_data(ds_train, "Train")
        _show_sample_from_ds_data(ds_val, "Validation")
        _show_sample_from_ds_data(ds_test, "Test")

        ''' Uncomment below to print tensor dimensions and data type '''
        # ds_train.element_spec

        ''' Prepare function for preparing the dataset for performance(batching, prefetching) '''

        return prepare_for_performance(ds_train, ds_val, ds_test)
    else:
        return ValueError


@gin.configurable
def path2dir(dataset_directory, images_train, images_test, csv_train_labels, csv_test_labels):
    """
    Purpose: To return all paths to directories that are to be used while loading a dataset

    Args:
        dataset_directory: path to directory od Dataset
        images_train: path to training images directory from Dataset directory
        images_test:  path to testing images directory from Dataset directory
        csv_train_labels: path to train.csv  directory from Dataset directory
        csv_test_labels: path to test.csv  directory from Dataset directory

    Returns: directory paths of training images,testing images,
             training labels (in csv), testing labels (in csv).

    """
    path_train_images = dataset_directory + images_train
    path_test_images = dataset_directory + images_test
    path_train_csv = dataset_directory + csv_train_labels
    path_test_csv = dataset_directory + csv_test_labels
    return path_train_images, path_test_images, path_train_csv, path_test_csv


def show_sample_image(files_dir):
    """
    Purpose: Displays an images randomly from a directory of images
    Args:
        files_dir: Path to the directory where the images are located.
    """
    list_train_files = glob.glob(files_dir + '/*.jpg')
    filename = list_train_files[random.randint(0, len(list_train_files))]
    img = plt.imread(filename)
    plt.imshow(img)
    plt.show()
    pass


def load_from_csv(file_dir, cols_used):
    """
    Purpose: To load csv files into a pandas dataframe,
             and replace labels if multiclass classifications is not preferred

    Args:
        file_dir: path where csv is located
        cols_used: columns to be considered while reading a csv to a pandas dataframe

    Returns: pandas data frame with mentioned columns in cols_used

    """

    # Load csv file into a pandas dataframe
    data_frame_from_csv = pd.read_csv(file_dir, usecols=cols_used, dtype=str)

    '''Code for assigning classes 0,1,2 to 0(Non proliferative) and 1(proliferative) '''
    '''comment the below code if you want to do multi class classification'''
    # Replacing dataframe columns with
    data_frame_from_csv.loc[(data_frame_from_csv[data_frame_from_csv.columns[1]] == '0') |
                            (data_frame_from_csv[data_frame_from_csv.columns[1]] == '1'),
                            data_frame_from_csv.columns[1]] = '0'

    data_frame_from_csv.loc[(data_frame_from_csv[data_frame_from_csv.columns[1]] == '2') |
                            (data_frame_from_csv[data_frame_from_csv.columns[1]] == '3') |
                            (data_frame_from_csv[data_frame_from_csv.columns[1]] == '4'),
                            data_frame_from_csv.columns[1]] = '1'

    return data_frame_from_csv


def _append_file_format_extension2name(df_names_labels):
    """
    Purpose: append file extenstion to the image name column in pandas dataframe

    Args:
        df_names_labels: pandas dataframe that contains Image names and corresponding labels

    Returns:

    """

    def _append_ext(fn):
        return fn + ".jpg"

    df_names_labels["Image name"] = df_names_labels["Image name"].apply(_append_ext)

    return df_names_labels


def _show_sample_from_df_iter(df_iter_test_data):
    """
    Purpose: To display sample image from data frame iterator(Its a method of ImageDataGenerator object),
        to check fetched image
    Args:
        df_iter_test_data: A dataframe iterator which is returned from .flow_from_dataframe method
    """
    # df_iter_test_data.next() returns a tuple of( batch of images, batch of labels)
    t_sample_image, t_sample_label = df_iter_test_data.next()

    # convert one numpy nd array from the fetched batch to a integer array for displaying image
    '''If images are not rescaled uncomment the below line'''
    # plt.imshow(t_sample_image[0].astype('uint8'))
    '''If images are  rescaled uncomment the below line'''
    plt.imshow(t_sample_image[0])

    # getting integer image label from one hot encoded label
    image_label = (np.where(t_sample_label[0] == 1))[0].tolist()[0]
    # plot image with integer label
    plt.title("Class of the Image is %d" % image_label)
    plt.show()


def _show_sample_from_ds_data(tf_ds, dataset_name):
    """
    Purpose : To display images in a grid of 9x9, from tensor flow dataset(returned using a
    tf.data.Dataset.from_generator()), to check fetched image from a sample batch(batch size should be grater than 9)
    Args: tf_ds:
    """
    plt.figure(figsize=(10, 10))
    plt.suptitle("Samples from augmented %s dataset" % dataset_name)
    for images, labels in tf_ds.take(1):

        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)

            '''If images are not rescaled uncomment the below code'''
            # plt.imshow(images[i].numpy().astype("uint8"))

            '''If images are  rescaled uncomment the below line'''
            plt.imshow(images[i])

            '''Uncomment below line for one hot coded labels'''
            plt.title("Class of image: %d " % labels[i])

            '''Uncomment below line for one hot coded labels'''
            # plt.title("Class of image: %d " % ((np.where(labels[i] == 1))[0].tolist()[0]))
            plt.axis("off")
    plt.show()


@gin.configurable
def _build_dataset(df_pandas_dataframe, directory_of_images,
                   image_generator, class_mode, subset_name, img_height, img_width, shuffle_val):
    """
    Purpose: To create a tensorflow data set from_generator using
             ImageDataGenerator(using the method flow_from_dataframe)

    Args:
        df_pandas_dataframe: pandas dataframe containing Image file names and
                             their respective labels in corresponding columns

        directory_of_images: path to where images of dataset to be built are located
        image_generator: ImageDataGenerator instance of keras
        class_mode:  For multiclass mention categorical,
                     for other options check https://keras.io/api/preprocessing/image/#flowfromdataframe-method
        subset_name: if validation split is mentioned for the respective ImageDataGenerator , then mention
                     subset name to be 'training' or 'validation'

    Returns:

    """

    dataframe_iterator = image_generator.flow_from_dataframe(df_pandas_dataframe,
                                                             directory=directory_of_images,
                                                             x_col=df_pandas_dataframe.columns[0],
                                                             y_col=df_pandas_dataframe.columns[1],
                                                             subset=subset_name,
                                                             seed=50,
                                                             target_size=(img_height, img_width),
                                                             batch_size=constants.N_BATCH_SIZE,
                                                             class_mode=class_mode, shuffle=shuffle_val)

    # uncomment the following code to visualize the sample image after the generator
    # _show_sample_from_df_iter(dataframe_iterator)

    # fetches a batch(batch size = constants.N_BATCH_SIZE) of images and labels
    images, labels = iter(dataframe_iterator.next())
    print(images.shape, labels.shape)

    ds_data = tf.data.Dataset.from_generator(lambda: dataframe_iterator,
                                             output_types=(tf.float32, tf.uint8),
                                             output_shapes=([None, images.shape[1], images.shape[2], 3], [None, ]))
    # (images.shape, labels.shape))

    return ds_data


@gin.configurable
def prepare_for_performance(ds_train, ds_val, ds_test, caching):
    """
    Purpose: To well shuffle and batch the data,
             then to prefetch the batch to be available to model as an input

    Args:
        caching:
        ds_test: test data set
        ds_val: validation data set(percentage of split from training data, mentioned in "constants.py")
        ds_train: training data set

    Returns: shuffled,batched, and prefetched

    """

    '''Prepare training dataset'''
    # ds_train = ds_train.map(crop2bb, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # cache will have a complete list of the elements in the dataset, and it will be used on all subsequent iterations
    if caching:
        ds_train = ds_train.cache()
    # shuffle and repeat
    # ds_train = ds_train.shuffle(constants.N_SHUFFLE_BUFFER)

    ds_train = ds_train.repeat(-1)

    # prefetch data
    ds_train = ds_train.prefetch(AUTOTUNE)

    '''Prepare validation dataset'''
    # cache will have a complete list of the elements in the dataset, and it will be used on all subsequent iterations
    if caching:
        ds_val = ds_val.cache()
    # Shuffling not needed for validation and testing data
    ds_val = ds_val.prefetch(AUTOTUNE)

    '''Prepare test dataset'''

    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(AUTOTUNE)

    return ds_train, ds_val, ds_test
