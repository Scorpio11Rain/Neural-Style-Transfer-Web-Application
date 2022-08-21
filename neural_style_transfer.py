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
OPTIMIZER = tf.optimizers.Adam(learning_rate=0.03)
STYLE_LAYERS = [
    ('block1_conv1', 1.0),
    ('block2_conv1', 0.8),
    ('block3_conv1', 0.7),
    ('block4_conv1', 0.2),
    ('block5_conv1', 0.1)]
CONTENT_LAYER = [('block5_conv4', 1)]


# load single image from BytesIO to tensor
def load_img_byte(buf):
    img = plt.imread(buf)
    img = Image.fromarray(img)
    width, height = img.size
    original_shape = (width, height)
    img = np.array(img.resize((IMG_SIZE, IMG_SIZE)))
    img = tf.constant(np.reshape(img, ((1,) + img.shape)))
    return img, original_shape


# to get orginal dimension, we need to change the order of width and height
def load_img_locally(filename):
    image = Image.open(filename)
    width, height = image.size
    original_shape = (width, height)
    image = np.array(image.resize((IMG_SIZE, IMG_SIZE)))
    image = tf.constant(np.reshape(image, ((1,) + image.shape)))

    return image, original_shape


def compute_content_cost(content_output, generated_output):
    a_C = content_output[-1]
    a_G = generated_output[-1]
    
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_C_unrolled = tf.reshape(a_C, shape=[m, n_H * n_W, n_C])
    a_G_unrolled = tf.reshape(a_G, shape=[m, n_H * n_W, n_C])

    J_content =  tf.reduce_sum(tf.square(a_C_unrolled - a_G_unrolled))/(4.0 * n_H * n_W * n_C)

    return J_content


# helper function
def compute_layer_style_cost(a_S, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    def gram_matrix(A):
        GA = tf.matmul(A, tf.transpose(A))
        return GA

    a_S = tf.transpose(tf.reshape(a_S, shape=[-1, n_C]))
    a_G = tf.transpose(tf.reshape(a_G, shape=[-1, n_C]))
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    J_style_layer = tf.reduce_sum(tf.square(GS - GG))/(4.0 *(( n_H * n_W * n_C)**2))
    
    return J_style_layer


def compute_style_cost(style_image_output, generated_image_output, STYLE_LAYERS=STYLE_LAYERS):
    J_style = 0
    a_S = style_image_output[1:]

    a_G = generated_image_output[1:]
    for i, weight in zip(range(len(a_S)), STYLE_LAYERS):  
        J_style_layer = compute_layer_style_cost(a_S[i], a_G[i])
        J_style += weight[1] * J_style_layer

    return J_style


@tf.function()
def total_cost(J_content, J_style, alpha = 10, beta = 40):
    J = alpha * J_content + beta * J_style
    return J


def load_vgg_model_outputs():
    vgg = tf.keras.applications.VGG19(include_top=False,input_shape=(IMG_SIZE, IMG_SIZE, 3),weights='imagenet')
    vgg.trainable = False  
    
    def get_layer_outputs(vgg, layer_names):
        outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]
        model = tf.keras.Model([vgg.input], outputs)
        return model  
    
    vgg_model_outputs = get_layer_outputs(vgg, STYLE_LAYERS + CONTENT_LAYER)
    return vgg_model_outputs


@tf.function()
def train_step(generated_image, a_S, a_C, vgg_model_outputs, alpha = 10, beta = 40):
    with tf.GradientTape() as tape:
        a_G = vgg_model_outputs(generated_image)

        J_style = compute_style_cost(a_S, a_G)

        J_content = compute_content_cost(a_C, a_G)

        J = total_cost(J_content, J_style,alpha = alpha, beta = beta)

    grad = tape.gradient(J, generated_image)

    OPTIMIZER.apply_gradients([(grad, generated_image)])
    generated_image.assign(tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0))
    return J


def stylize(content_image, style_image, epochs):
    generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))

    vgg_model_outputs = load_vgg_model_outputs()

    preprocessed_style =  tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
    a_S = vgg_model_outputs(preprocessed_style)

    preprocessed_content =  tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
    a_C = vgg_model_outputs(preprocessed_content)
    
    for i in range(epochs):
        train_step(generated_image,a_S, a_C, vgg_model_outputs)

    def tensor_to_array(tensor):
        tensor = tensor * 255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return tensor
    
    return tensor_to_array(generated_image)




def save_image(img, img_shape):
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize(img_shape)
    buffer = BytesIO()
    pil_img.save(buffer, format = "PNG")
    byte_img = buffer.getvalue()
    return byte_img