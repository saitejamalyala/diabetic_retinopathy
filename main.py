import gin
from absl import app, flags
from matplotlib import pyplot as plt
import tensorflow as tf
import datetime
import os

'''User Defined Imports'''
import constants
from evaluation import eval
from models.transfer_learning_architecture import transfer_learning
from models.architecture import vgg_base_3custom
from input_pipeline import datasets, datasets2

''' ##############################################################################################################  '''
# Change the Flags below to set the program to respective Mode
FLAGS = flags.FLAGS

flags.DEFINE_boolean('train', False, 'Specify whether to train or evaluate a model.')

flags.DEFINE_boolean('ds2', True, 'Specify whether to use alternate data pipeline')

flags.DEFINE_boolean('hparam_tune', False, 'Specify if its hyper param tuning.')

flags.DEFINE_boolean('Transfer_learning', False, 'to use transfer learning based model, \
                                                        train flag must be set to true to fine tune pretrained model')


# -------------------------------------------------------------------------------------------------------------------- #

def main(argv):
    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])

    if FLAGS.hparam_tune:
        # For hyper parameter Tuning choose this mode
        from hyper_parameter_tuning.hparam_tuning import run_hparam_tuning
        run_hparam_tuning()

    else:
        if FLAGS.ds2:
            # setup pipeline without image data generator
            ds_train, ds_val, ds_test = datasets2.load_data()

            if FLAGS.Transfer_learning:
                # for finetuning pretrained model
                epochs = constants.H_TRANSFER_LEARNING_EPOCHS
                model = transfer_learning((256, 256, 3))
            else:
                # For using Custom model
                epochs = constants.H_EPOCHS
                model = vgg_base_3custom((256, 256, 3))

        else:
            # use pipeline using image data generator
            ds_train, ds_val, ds_test = datasets.load()
            if FLAGS.Transfer_learning:
                epochs = constants.H_TRANSFER_LEARNING_EPOCHS
                model = transfer_learning((256, 256, 3))
            else:
                epochs = constants.H_EPOCHS
                model = vgg_base_3custom((256, 256, 3))

        opt = tf.optimizers.Adam(constants.H_LEARNING_RATE, name='ADAM')

        if FLAGS.train:

            # Build and compile the model
            model.build((constants.N_BATCH_SIZE, constants.ip_shape[0], constants.ip_shape[1], 3))

            model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'], )
            print(model.summary())

            # tensor board call back
            if not os.path.isdir(constants.dir_fit):
                os.makedirs(constants.dir_fit)
            log_dir = os.path.join(constants.dir_fit, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            tensorboard_callbk = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,
                                                                write_graph=True,
                                                                write_images=True,
                                                                update_freq='epoch',
                                                                # profile_batch=2,
                                                                embeddings_freq=1)

            # Checkpoint call back
            cpt_dir = os.path.join(constants.dir_cpts, datetime.datetime.now().strftime("%Y%m%d-%H%M"))
            if not os.path.isdir(cpt_dir):
                os.makedirs(cpt_dir)
            print(cpt_dir)
            checkpoint_dir = os.path.join(cpt_dir, 'epochs:{epoch:03d}-val_accuracy:{val_accuracy:.3f}.h5')
            # check point to save the model based on improving validation accuracy
            checkpoint_callbk = tf.keras.callbacks.ModelCheckpoint(checkpoint_dir,
                                                                   monitor='val_accuracy',
                                                                   verbose=1,
                                                                   save_best_only=False,
                                                                   mode='max', save_weights_only=False,
                                                                   save_freq='epoch')
            # csv  call back, if dir doesnt exist create directory
            if not os.path.isdir(constants.dir_csv):
                os.makedirs(constants.dir_csv)
            log_file_name = os.path.join(constants.dir_csv,
                                         (datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'))
            csv_callbk = tf.keras.callbacks.CSVLogger(log_file_name, separator=',', append=True)

            callbacks_list = [checkpoint_callbk, tensorboard_callbk, csv_callbk]

            # Training the model and saving it using checkpoint call back
            history_model = model.fit(ds_train, verbose=1, epochs=int(epochs / 2), batch_size=constants.N_BATCH_SIZE,
                                      validation_data=ds_val,
                                      callbacks=callbacks_list)
            # training the saved model for rest of the epochs
            history_model = model.fit(ds_train, verbose=1, initial_epoch=int(epochs / 2), epochs=epochs,
                                      batch_size=constants.N_BATCH_SIZE,
                                      validation_data=ds_val,
                                      callbacks=callbacks_list)

            # save final model
            if not os.path.isdir(constants.WEIGHTS_PATH):
                os.makedirs(constants.WEIGHTS_PATH)
            model_save_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            model_name = model_save_time + '_' + model.optimizer.get_config()['name'] + '_epochs_' + str(epochs) + '.h5'
            model_save_path = os.path.join(constants.WEIGHTS_PATH, model_name)
            print(model_save_path)
            try:
                _ = os.stat(constants.WEIGHTS_PATH)
                model.save(model_save_path)
            except NotADirectoryError:
                raise

            # plot final training data, for runtime progress look at tensor board log
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.plot(history_model.history["loss"])
            plt.plot(history_model.history["val_loss"])
            plt.legend(["loss", "val_loss"])
            # plt.xticks(range(constants.H_EPOCHS))
            plt.xlabel("epochs")
            plt.title("Train and val loss")

            plt.subplot(1, 2, 2)
            plt.plot(history_model.history["accuracy"])
            plt.plot(history_model.history["val_accuracy"])
            plt.legend(["accuracy", "val_accuracy"])
            plt.title("Train and Val acc")
            plt.show()

            '''
            test_history = model.evaluate(ds_test,
                                          batch_size=constants.N_BATCH_SIZE,
                                          verbose=1, steps=4)

            '''

            eval.evaluate(model=model, ds_test=ds_test, opt=opt, is_training=FLAGS.train, SAVE_RESULT=True,
                          checkpoint_path=None)

        else:

            # Load checkpoint model to evaluate
            check_point_path = constants.trained_model_name

            # check_point_path = 'weights/20201222-220802_ADAM_epochs_100_test_acc_78.h5'
            eval.evaluate(model=model, ds_test=ds_test, opt=opt, is_training=FLAGS.train, SAVE_RESULT=True,
                          checkpoint_path=check_point_path)


if __name__ == "__main__":
    app.run(main)
