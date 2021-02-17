import gin
import tensorflow as tf
import pandas as pd

@gin.configurable
def preprocess(image):

    """
    PURPOSE: Dataset preprocessing: cropping and resizing

    Args:
        image: image to be preprocessed
    """
    image_cbb = tf.image.crop_to_bounding_box(image, 0, 15, 256, 209)
    image_resized = tf.image.resize(image_cbb, (256, 256))

    return image_resized


def resampling(df_imbalanced, frac=1):
    """

    Args:
        df_imbalanced: imbalanced data frame of paths and labels
        frac: frac argument in dataframe.sample(method)

    Returns: Resampled Dataframe

    """
    df_imbalanced = df_imbalanced.astype({'Retinopathy grade': int})
    df_minority = df_imbalanced[df_imbalanced['Retinopathy grade'] == 0]
    df_majority = df_imbalanced[df_imbalanced['Retinopathy grade'] == 1]
    # Calculate the imbalance of data, minority class frequency- majority class frequency
    difference = len(df_majority) - len(df_minority)
    # print(difference)
    df_sampled_from_minority = df_minority.sample(n=difference)
    # print(train_df_new_0.head())

    # concatenate the minority class, majority class and newly sampled class from minority
    df_balanced_data = pd.concat([df_minority, df_majority, df_sampled_from_minority], axis=0)
    # print(len(train_df))

    # shuffle the resampled data
    df_balanced_data = df_balanced_data.sample(frac=frac)
    # convert the labels to strings to be accepted by flow from dataframe
    df_balanced_data = df_balanced_data.astype({'Retinopathy grade': str})

    return df_balanced_data
