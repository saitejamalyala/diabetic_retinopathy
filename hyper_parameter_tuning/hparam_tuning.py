from tensorboard.plugins.hparams import api as hp
import constants
import tensorflow as tf
from input_pipeline import datasets2
import datetime
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

# uncomment below to tune on further parameters
'''
HP_CNN_DROPOUT = hp.HParam("fcn_dropout",display_name="CONV2D NW dropout",
                         description="Dropout rate for conv subnet.",
                          hp.RealInterval(0.1, 0.2))

 HP_FC_DROPOUT = hp.HParam("fc_dropout",display_name="f.c. dropout",
                          description="Dropout rate for fully connected subnet.",
                          hp.RealInterval(0.2, 0.5))
'''
HP_EPOCHS = hp.HParam("epochs", hp.Discrete([100, 140]),
                      description="Number of epoch to run")
HP_NEURONS = hp.HParam("num_Dense_layer_neurons", hp.Discrete([128, 256]),
                       description="Neurons per dense layer")
HP_STRIDE = hp.HParam("stride_in_first_layer", hp.Discrete([2, 1]),
                      description="Value of stride in frist convolutional layer")
HP_L_RATE = hp.HParam("learning_rate", hp.Discrete([0.0001, 0.00001]),
                      description="Learning rate")

HP_METRIC = hp.Metric(constants.METRICS_ACCURACY, display_name='Accuracy')

# creating logs for different hyper-parameters
with tf.summary.create_file_writer('hp_log_dir/hparam_tuning').as_default():
    hp.hparams_config(
        hparams=[HP_NEURONS, HP_EPOCHS, HP_L_RATE, HP_STRIDE],
        metrics=[HP_METRIC],
    )


def run(run_dir, run_name, hparams, gen_train, gen_valid, gen_test):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        accuracy = train_test_model(gen_train, gen_valid, gen_test, hparams, run_name)
        tf.summary.scalar(constants.METRICS_ACCURACY, accuracy, step=1)


###
def train_test_model(gen_train, gen_valid, gen_test, hparams, run_name):
    inputs = tf.keras.layers.Input(constants.ip_shape)

    out = tf.keras.layers.Conv2D(8, 3, hparams[HP_STRIDE], padding='same', activation=tf.nn.relu)(inputs)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.MaxPool2D((3, 3))(out)

    out = tf.keras.layers.Conv2D(16, 3, padding='same', activation=tf.nn.relu)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)

    out = tf.keras.layers.Conv2D(32, 3, padding='same', activation=tf.nn.relu)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)

    out = tf.keras.layers.Conv2D(128, 3, padding='same', activation=tf.nn.relu)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)

    out = tf.keras.layers.Dropout(0.3)(out)
    out = tf.keras.layers.Flatten()(out)

    l2_reg = tf.keras.regularizers.l2(0.001)
    # l1_l2_reg = tf.keras.regularizers.L1L2(l1=0.001,l2=0.001)
    # tried 512 without following dropout of 0.3
    out = tf.keras.layers.Dense(hparams[HP_NEURONS], activation='linear',
                                kernel_regularizer=l2_reg)(out)
    out = tf.keras.activations.relu(out)
    out = tf.keras.layers.Dropout(0.5)(out)
    # out = tf.keras.layers.Dense(32, activation=tf.nn.relu)(out)
    final_out = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(out)

    hp_model = tf.keras.Model(inputs=inputs, outputs=final_out, name="HP_tuning_DR_model")

    opt = tf.optimizers.Adam(hparams[HP_L_RATE], name='ADAM')

    hp_model.build((constants.N_BATCH_SIZE, constants.ip_shape))

    hp_model.compile(optimizer=opt,
                     loss=tf.keras.losses.sparse_categorical_crossentropy,
                     metrics=constants.METRICS_ACCURACY)
    print(hp_model.summary())
    hp_model.fit(gen_train, batch_size=constants.N_BATCH_SIZE,
                 epochs=hparams[HP_EPOCHS], verbose=1,
                 steps_per_epoch=((constants.N_TRAIN_SIZE_POST_AUG // constants.N_BATCH_SIZE) + 1),
                 validation_data=gen_valid,
                 validation_steps=(constants.N_VALID_SIZE_POST_AUG // constants.N_BATCH_SIZE) + 1,
                 callbacks=call_backs(hparams, run_name))

    loss, accuracy = hp_model.evaluate(gen_test, batch_size=constants.N_BATCH_SIZE,
                                       verbose=1,
                                       steps=(constants.N_TESTING_SET_COUNT // constants.N_BATCH_SIZE + 1),
                                       )
    save_test_results(gen_test, hp_model, run_name)

    return accuracy


def call_backs(hparams, run_name):
    # tensorboard call back
    log_dir = './hp_log_dir/fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + run_name
    tensorboard_callbk = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,
                                                        write_graph=True,
                                                        write_images=True,
                                                        update_freq='epoch',
                                                        profile_batch=2,
                                                        embeddings_freq=1)

    # model checkpoint call back
    cpt_path = "./hp_log_dir/cpts/" + run_name + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + \
               "epochs:{epoch:03d}-val_accuracy:{val_accuracy:.3f}.h5"
    # cpt_path = "./hp_log_dir/cpts/" + run_name + "_" + "cp-epochs:{epoch:03d}-val_accuracy:{val_accuracy:.3f}.ckpt"
    # check point to save the model based on improving validation accuracy
    checkpoint_callbk = tf.keras.callbacks.ModelCheckpoint(cpt_path,
                                                           monitor='val_accuracy',
                                                           verbose=1,
                                                           save_best_only=False,
                                                           mode='max', save_weights_only=True,
                                                           save_freq='epoch')
    # csv logger call back
    log_file_name = './hp_log_dir/csv_log/' + run_name + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_callbk = tf.keras.callbacks.CSVLogger(log_file_name, separator=',', append=True)

    # keras callback for hzper param
    hp_log_dir = './hp_log_dir/hparam_tuning/' + run_name
    hp_callbk = hp.KerasCallback(hp_log_dir, hparams)  # log hparams

    callbacks_list = [checkpoint_callbk, tensorboard_callbk, csv_callbk, hp_callbk]
    return callbacks_list


def run_hparam_tuning():
    session_num = 0
    # Get data from datasets.py or datasets2.py
    # gen_train, gen_valid, gen_test = datasets.load()
    gen_train, gen_valid, gen_test = datasets2.load_data()

    for num_Dense_layer_neurons in HP_NEURONS.domain.values:
        for epochs in HP_EPOCHS.domain.values:
            for learning_rate in HP_L_RATE.domain.values:
                for stride_in_first_layer in HP_STRIDE.domain.values:
                    hparams = {
                        HP_NEURONS: num_Dense_layer_neurons,
                        HP_EPOCHS: epochs,
                        HP_L_RATE: learning_rate,
                        HP_STRIDE: stride_in_first_layer,
                    }
                    run_name = "run-%d" % session_num
                    print('--- Starting trial: %s' % run_name)
                    print({h.name: hparams[h] for h in hparams})
                    run('hp_log_dir/hparam_tuning/' + run_name, run_name, hparams, gen_train, gen_valid, gen_test)
                    session_num += 1


def save_test_results(gen_test, saved_model, run_name):
    true_labels = []
    for data, labels in (gen_test.take((constants.N_TESTING_SET_COUNT // constants.N_BATCH_SIZE) + 1)):
        true_labels.extend(labels.numpy().tolist())

    print(true_labels)
    # saved_model = tf.keras.models.load_model('20201215-190832SGD_100.h5')

    test_model = saved_model.evaluate(gen_test,
                                      batch_size=constants.N_BATCH_SIZE,
                                      verbose=1, steps=4)
    print(test_model)
    y_pred = saved_model.predict(gen_test,
                                 batch_size=constants.N_BATCH_SIZE,
                                 steps=(constants.N_TESTING_SET_COUNT // constants.N_BATCH_SIZE) + 1,
                                 verbose=1)

    y_pred = np.argmax(y_pred, axis=1)
    print(y_pred)
    print('Confusion Matrix')
    print(confusion_matrix(true_labels, y_pred))
    plt.figure()
    cm_plot = sns.heatmap(confusion_matrix(true_labels, y_pred), annot=True)

    cm_fig = cm_plot.get_figure()
    cm_fig.savefig("./hp_log_dir/results/%s_.png" % run_name)
    print('Classification Report')
    target_names = ['NPDR', 'PDR']
    cr_data = classification_report(true_labels, y_pred, target_names=target_names, output_dict=True)
    print(cr_data)
    df_cr_data = pd.DataFrame(cr_data).transpose()
    df_cr_data.to_csv("./hp_log_dir/results/%s_.csv" % run_name)
