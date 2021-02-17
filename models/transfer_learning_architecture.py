from tensorflow.keras.applications import ResNet50V2
import tensorflow as tf
import tensorflow.keras as keras


def transfer_learning(input_shape):
    """
    PURPOSE : To return model, built on a pretrained model(ResNet50V2)
    Args:
        input_shape: shape of the the input to the model

    Returns: Keras model

    """
    base_model = ResNet50V2(include_top=False, input_shape=(224, 224, 3), pooling='avg', weights='imagenet')

    # Freeze the layers except the last 12 layers (which contains few sets of Conv layers and batch normalization
    # layers)
    count_layers = 0
    for layer in base_model.layers[:-12]:
        layer.trainable = False
        count_layers = count_layers + 1
    print(count_layers, "Number of layers in Resnet50")

    # Check the trainable status of the individual layers
    for layer in base_model.layers:
        print(layer, layer.trainable)
    base_model.summary()

    # Keras input layer
    inputs = keras.Input(shape=input_shape)

    # preprocessing layer to resize image to 224*224, as Resnet input layer accepts 224,224,3
    r_input = keras.layers.experimental.preprocessing.Resizing(224, 224)(inputs)

    out = base_model(r_input)

    out = keras.layers.Dense(16, activation=tf.nn.relu,kernel_regularizer=keras.regularizers.l1(0.0001))(out)

    out = keras.layers.Dropout(0.6)(out)

    out = keras.layers.Dense(2, activation=tf.nn.softmax)(out)

    model = keras.Model(inputs, out)

    # Model Summary
    model.summary()
    return model
