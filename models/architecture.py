import tensorflow as tf


def vgg_base_3custom(ip_shape):

    """Testing random search param

    Args:
        ip_shape: shape of the input to model
    return : Keras model
    """

    # Create the model

    inputs = tf.keras.layers.Input(ip_shape)

    out = tf.keras.layers.Conv2D(8, 3, 2, padding='same', activation=tf.nn.relu)(inputs)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.MaxPool2D((3, 3))(out)

    out = tf.keras.layers.Conv2D(16, 3, padding='same', activation=tf.nn.relu)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)

    ''' 
    out = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)(inputs)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)
    out = tf.keras.layers.Dropout(0.25)(out)
    '''

    out = tf.keras.layers.Conv2D(32, 3,  padding='same', activation=tf.nn.relu)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)

    out = tf.keras.layers.Conv2D(128, 3,  padding='same', activation=tf.nn.relu)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)

    out = tf.keras.layers.Dropout(0.3)(out)
    out = tf.keras.layers.Flatten()(out)

    l2_reg = tf.keras.regularizers.l2(0.001)
    # l1_l2_reg = tf.keras.regularizers.L1L2(l1=0.001,l2=0.001)
    out = tf.keras.layers.Dense(128, activation='linear',
                                kernel_regularizer=l2_reg)(out)
    out = tf.keras.activations.relu(out)
    out = tf.keras.layers.Dropout(0.5)(out)
    # out = tf.keras.layers.Dense(32, activation=tf.nn.relu)(out)
    out = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(out)

    # Show a summary of the model. Check the number of trainable parameters
    # model.summary()

    model = tf.keras.Model(inputs=inputs, outputs=out, name='DBR_model')
    return model
