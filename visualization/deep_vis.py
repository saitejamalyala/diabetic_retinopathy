import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import cv2

'''User Defined Imports'''
import constants
from input_pipeline import datasets2

# get train,test,valid datasets
_, _, ds_test = datasets2.load_data()

# path to test image
f_path = 'C:/Users/Teja/Documents/_INFOTECH/sem5/DL_lab/IDRID_dataset/images/test/IDRiD_033.jpg'

# path to the saved model 
saved_model = tf.keras.models.load_model('weights/20201222-220802_ADAM_epochs_100_test_acc_78.h5')

# compile the loaded keras model
saved_model.compile(optimizer=tf.keras.optimizers.Adam(constants.H_LEARNING_RATE),
                    loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'], )
# build the compiled keras model with input shape = [batchsize,image shape]
saved_model.build((32, 256, 256, 3))
print(saved_model.summary())

img_size = (constants.ip_shape[0], constants.ip_shape[1])

# get the last convolution layer to perform Grad CAM
last_conv_layer_name = "conv2d_3"
# list of all layers after the selected convolution layer till classification
classifier_layer_names = ["batch_normalization_3", "max_pooling2d_3", "dropout", "flatten", "dense", "tf_op_layer_Relu",
                          "dropout_1", "dense_1"]


def _get_img_array(f_path: str):
    '''
    NOTE: `img` is a PIL image of size 256x256
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (256, 256, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 256, 256, 3)

    Args:
        f_path (str): path to read the located Image
    '''
    image_string = tf.io.read_file(f_path)
    print(type(image_string))
    image = tf.io.decode_jpeg(image_string, channels=3)
    image = tf.image.crop_to_bounding_box(image, 0, 266, 2848, 3426)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, [256, 256])
    img_arr = np.expand_dims(image, axis=0)

    return img_arr


def _make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):
    """

    Args:
        img_array: image for which grad CAM will be performed
        model: TRained deep neural network
        last_conv_layer_name:
        classifier_layer_names:

    Returns:

    """
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)
    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)
    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]
    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]
    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)
    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap


img_array = _get_img_array(f_path)
# Print what the top predicted class is
preds = saved_model.predict(img_array)
# print("Predicted:", decode_predictions(preds, top=1)[0])
# Generate class activation heatmap
cam = _make_gradcam_heatmap(img_array, saved_model, last_conv_layer_name, classifier_layer_names)

# Display heatmap
img = keras.preprocessing.image.load_img(f_path)
img = img.crop(box=(266, 0, 3692, 2848))
img = img.resize((256, 256))

# resize heatmap, then convert it to 3 channel (apply colormap)
cam_res = cv2.resize(cam, (256, 256))
heat_map = cv2.applyColorMap(np.uint8(255 * cam_res), cv2.COLORMAP_JET)
added_map = cv2.addWeighted(cv2.cvtColor(np.asarray(img).astype('uint8'), cv2.COLOR_RGB2BGR), 0.7, heat_map, 0.4, 0)


# Plot image, gradcam output and gradcam overlay
plt.figure(1)
plt.subplot(1, 3, 1)
plt.axis("off")
plt.imshow(img)
plt.subplot(1, 3, 2)
plt.axis("off")
plt.imshow(heat_map)
plt.subplot(1, 3, 3)
plt.axis("off")
plt.imshow(added_map)

# np.resize(np.squeeze(img_array,axis=0),[16,16]))
plt.show()
plt.figure(2)
overlay_map = np.float32(heat_map) + np.float32(img) * 0.4  # everlay heatmap onto the image
overlay_map = 255 * overlay_map / np.max(overlay_map)
overlay_map = np.uint8(overlay_map)
plt.imshow(overlay_map)
plt.show()

