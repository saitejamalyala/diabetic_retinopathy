import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import os

'''User Defined Imports'''
import constants

def _classification_report_csv(report, conf_mat):
    """
    Args:
        report: classification report (type dict)
        conf_mat: Confusion Matrix

    Returns:
        object: None
    """
    dataframe = pd.DataFrame.from_dict(report)
    if os.path.isdir(constants.results_PATH):
        conf_mat.savefig(constants.results_PATH + "confusionmatrix.png")
        dataframe.to_csv(constants.results_PATH + 'classification_report.csv', index=False)
    else:
        try:
            os.makedirs(constants.results_PATH)
            conf_mat.savefig(constants.results_PATH + "confusionmatrix.png")
            dataframe.to_csv(constants.results_PATH + 'classification_report.csv', index=False)
        except FileExistsError:
            pass
        except OSError:
            raise


def evaluate(model, ds_test, opt, is_training, SAVE_RESULT=True, checkpoint_path=None):
    """
    PURPOSE: To (evaluate the model and predict on test dataset). Also to calculate confusion matrix and
             classification report and if mentioned save the results.
    Args:
        model: model instance
        ds_test: testing data set
        opt: optimizer
        is_training: if in training mode will be True
        SAVE_RESULT: To save the results (confusion matrix and classification report)
        checkpoint_path: path to check pointed model

    """
    true_labels = []
    for data, labels in (ds_test.take((constants.N_TESTING_SET_COUNT // constants.N_BATCH_SIZE) + 1)):
        true_labels.extend(labels.numpy().tolist())

    print('\n True labels:\n', true_labels)

    if is_training:
        saved_model = model
    else:
        try:
            _ = os.stat(checkpoint_path)
            # if os.path.isfile(checkpoint_path):
            print(os.path.sep, type(os.path.sep), type(checkpoint_path))
            print("Loading Checkpoint model {}".format(checkpoint_path.split(os.sep)[-1]))
            # For loading weights use loadedmodel.load_weights(checkpoint)
            saved_model = tf.keras.models.load_model(checkpoint_path, compile=False)
            # saved_model = model.load_weights(checkpoint_path)

            # Compile the model
            saved_model.compile(optimizer=tf.keras.optimizers.Adam(constants.H_LEARNING_RATE),
                                loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            print(saved_model.summary())

        except FileNotFoundError:
            raise

    # Evaluate the model
    print("\nEvaluating on test Dataset.....\n")
    test_model = saved_model.evaluate(ds_test,
                                      batch_size=constants.N_BATCH_SIZE,
                                      steps=(constants.N_TESTING_SET_COUNT // constants.N_BATCH_SIZE) + 1,
                                      verbose=1)
    # print(test_model)
    # Predict to calculate
    print("\nPredicting on test Dataset.....\n")
    y_pred = saved_model.predict(ds_test,
                                 batch_size=constants.N_BATCH_SIZE,
                                 steps=(constants.N_TESTING_SET_COUNT // constants.N_BATCH_SIZE) + 1,
                                 verbose=1)

    y_pred = np.argmax(y_pred, axis=1)
    print('\n Predicted labels:\n', y_pred)

    # y_true = np.asarray(y_true).astype('int32')

    # Confusion Matrix
    print('\n Confusion Matrix:\n')
    print(confusion_matrix(true_labels, y_pred))

    target_names = ['NRDR', 'RDR']
    plt.figure()
    sns.set(font_scale=1.8)
    cm_plot = sns.heatmap(confusion_matrix(true_labels, y_pred), annot=True, cbar=True,
                          xticklabels=target_names, yticklabels=target_names, annot_kws={"size": 65})

    cm_fig = cm_plot.get_figure()
    # plt.show()

    # Save classification report and confusion matrix to results folder
    if SAVE_RESULT:
        cr = classification_report(true_labels, y_pred, target_names=target_names, output_dict=True)
        _classification_report_csv(cr, cm_fig)

    # Classification report
    cr = classification_report(true_labels, y_pred, target_names=target_names)
    print('Classification Report:\n')
    print("\n", cr, "\n")

    return
