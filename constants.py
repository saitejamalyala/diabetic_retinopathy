import os

"""#################### Used in datasets.py #################### """
N_TRAINING_SET_COUNT = 413

VALIDATION_PERCENT = 0.2
TRAINING_PERCENT = 1 - VALIDATION_PERCENT

# 413-80 = 333 = training samples after validation split
# after augmentation 518-103=415 training  samples
N_VALIDATION_SAMPLES = int(N_TRAINING_SET_COUNT * VALIDATION_PERCENT)

N_TRAINING_SAMPLES = int(N_TRAINING_SET_COUNT * (1.0 - VALIDATION_PERCENT))

N_SHUFFLE_BUFFER = N_TRAINING_SAMPLES

N_TESTING_SET_COUNT = 103

COLUMN_LABELS = ['Image name', 'Retinopathy grade']

N_BATCH_SIZE = 32

data_dir = './IDRID_dataset'

METRICS_ACCURACY = 'accuracy'

""" #################### -------------------------------------------------------------- ###################### """

""" ##################### used in main.py #########################"""
# Hyper Parameters
H_LEARNING_RATE = 0.0001
H_EPOCHS = 2  # 100
H_TRANSFER_LEARNING_EPOCHS = 1  # 15
ip_shape = (256, 256, 3)
ip_shape_vgg = (224, 224, 3)

N_DATA_SET_SIZE_POST_AUG = 495
N_TRAIN_SIZE_POST_AUG = 412
N_VALID_SIZE_POST_AUG = 83


# Directories used
dir_all_logs = 'log_dir'
dir_csv = os.path.join(dir_all_logs, 'csv_log')
dir_fit = os.path.join(dir_all_logs, 'fit')
dir_cpts = os.path.join(dir_all_logs, 'cpts')

# path to store the trained models
WEIGHTS_PATH = './weights'

# trained model name or saved model(full model checkpoint) checkpoint name
trained_model_name = 'weights/fullmodel_tl_82_acc.h5'

""" #################### -------------------------------------------------------------- ###################### """

""" #################### used in eval.py ####################"""
# path to save results
results_PATH = './results/'

""" #################### used in datasets2.py ############## """
# paths for building tensorflow dataset
path_train_img = os.path.join(data_dir, 'images', 'train')
path_test_img = os.path.join(data_dir, 'images', 'test')
path_train_labels = os.path.join(data_dir, 'labels', 'train.csv')
path_test_labels = os.path.join(data_dir, 'labels', 'test.csv')

""" #################### -------------------------------------------------------------- ###################### """
