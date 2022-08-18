import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import VGG19
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from io import BytesIO


IMG_SIZE = 400
STYLE_WEIGHTS = 0.8
CONTENT_WEIGHTS = 0.2
OPTIMIZER = tf.optimizers.Adam(learning_rate=0.01, beta_1=0.99, epsilon=1e-1)

def gram_matrix(input_tensor):
    input_shape = tf.shape(input_tensor)
    ts_mul = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
    gram = tf.expand_dims(ts_mul, axis = 0)
    div = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    result = gram/div
    return result



def load_vgg_model():
    vgg = VGG19(include_top = False, input_shape = (IMG_SIZE, IMG_SIZE, 3), weights="imagenet")
    vgg.trainable = False
    content_layers = ['block4_conv2']
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

    input_layer = [vgg.input]
    content_out = vgg.get_layer(content_layers[0]).output
    style_out = [vgg.get_layer(style_layer).output for style_layer in style_layers]
    gram_style_out = [gram_matrix(output_) for output_ in style_out]
    output_layer = [content_out, gram_style_out]

    model = Model(input_layer, output_layer)

    return model

# to get orginal dimension, we need to change the order of width and height
def load_img_locally(filename):
    image = cv2.imread(filename)
    original_shape = (image.shape[1], image.shape[0])
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, original_shape

# load single image from BytesIO to tensor
def load_img_byte(buf):
    img = plt.imread(buf)
    original_shape = (img.shape[1], img.shape[0])
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img, original_shape

def calculate_loss(style_out, content_out, style_target, content_target):
    content_loss = tf.reduce_mean((content_out - content_target)**2)
    style_loss = tf.add_n([tf.reduce_mean((output - target)**2) for output, target in zip(style_out, style_target)])
    total_loss = STYLE_WEIGHTS * style_loss  + CONTENT_WEIGHTS * content_loss
    return total_loss


def stylize(content_image, style_image, model, epochs):
    content_target = model(np.array([content_image * 255]))[0]
    style_target = model(np.array([style_image * 255]))[1]

    image = tf.Variable([content_image])

    for _ in range(epochs):
        with tf.GradientTape() as tape:
            output = model(image * 255)
            loss = calculate_loss(output[1], output[0], style_target, content_target)
        gradient = tape.gradient(loss, image)
        OPTIMIZER.apply_gradients([(gradient, image)])
        image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))

    final_output = np.array(image*255, dtype=np.uint8)
    if np.ndim(final_output)>3:
         final_output= final_output[0]

    return final_output


def save_image(img, img_shape):
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize(img_shape)
    buffer = BytesIO()
    pil_img.save(buffer, format = "PNG")
    byte_img = buffer.getvalue()
    return byte_img

 


