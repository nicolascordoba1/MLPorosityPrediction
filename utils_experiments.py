import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np 
import tensorflow as tf
from keras.layers import Activation, Conv2D, Conv2DTranspose, Lambda, Input, Concatenate, BatchNormalization, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


def robust_scaling(data):
    median = np.median(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    return (data - median) / iqr
  
def unnormalize_robust(data_normalized, median_val, iqr_val):
    """
    Reverts the robust scaling normalization.

    Parameters:
    data_normalized (numpy array): The normalized data.
    median_val (float): The original median value before normalization.
    iqr_val (float): The original IQR (Interquartile Range) before normalization.

    Returns:
    numpy array: The unnormalized data.
    """
    return data_normalized * iqr_val + median_val

def scale_to_range(data, min_val=-1, max_val=1):
    """
    Scale data to a range (default [-1, 1]) using min-max normalization.
    
    Args:
        data: numpy array or list to be scaled
        min_val: minimum value of desired range (default -1)
        max_val: maximum value of desired range (default 1)
    
    Returns:
        scaled data in the specified range
    """
    data_min = np.min(data)
    data_max = np.max(data)
    
    # Prevent division by zero if all values are the same
    if data_max == data_min:
        return np.zeros_like(data)
    
    # First normalize to [0, 1]
    normalized = (data - data_min) / (data_max - data_min)
    
    # Then scale to [min_val, max_val]
    scaled = normalized * (max_val - min_val) + min_val
    
    return scaled

def unscale_from_range(scaled_data, original_min, original_max, min_val=-1, max_val=1):
    """
    Reverse the scaling operation to get back original values.
    
    Args:
        scaled_data: numpy array or list of scaled values
        original_min: minimum value from original dataset
        original_max: maximum value from original dataset
        min_val: minimum value of current range (default -1)
        max_val: maximum value of current range (default 1)
    
    Returns:
        data in original scale
    """
    # First normalize back to [0, 1]
    normalized = (scaled_data - min_val) / (max_val - min_val)
    
    # Then scale back to original range
    original = normalized * (original_max - original_min) + original_min
    
    return original

def visualizacion_resultados(history):

    train_acc = history.history['r2_score']
    train_loss = history.history['loss']
    train_lr = history.history['learning_rate']
    val_acc = history.history['val_r2_score']
    val_loss = history.history['val_loss']
    
    epochs = [i for i in range(len(train_acc))]

    fig, ax = plt.subplots(1,3)
    fig.set_size_inches(16,7)

    ax[0].plot(epochs, train_acc, 'go-', label='accuracy-train')
    ax[0].plot(epochs, val_acc, 'ro-', label='accuracy-val')
    ax[0].set_title('Accuracy train')
    ax[0].legend()
    ax[0].set_xlabel('epochs')
    ax[0].set_ylabel('accuracy')
    ax[0].set_ylim(-1,1)

    ax[1].plot(epochs, train_loss, 'go-', label='loss-train')
    ax[1].plot(epochs, val_loss, 'ro-', label='loss-val')
    ax[1].set_title('Loss train')
    ax[1].legend()
    ax[1].set_xlabel('epochs')
    ax[1].set_ylabel('loss')
    
    ax[2].plot(epochs, train_lr, 'go-', label='lr-train')
    ax[2].set_title('Learning Rate train')
    ax[2].legend()
    ax[2].set_xlabel('epochs')
    ax[2].set_ylabel('Learning Rate')
    
    fig.savefig('./training.png')

    plt.show()

def plot_nfe_experiment(coords_train, coords_val,seis_mature_normalized, porcentaje_entrenamiento, porcentaje_validacion):
    offset = np.array([13, 1300])

    coords_train = coords_train + offset
    coords_val = coords_val + offset
    
    # Extract x and y coordinates
    x_val = coords_val[:, 0]
    y_val = coords_val[:, 1]

    x_train = coords_train[:, 0]
    y_train = coords_train[:, 1]

    # Define the rectangle parameters
    rect_x_seismic = 13
    rect_y_seismic = 930
    rect_width_seismic = 195 - rect_x_seismic
    rect_height_seismic = 2140 - rect_y_seismic

    rect_x_mature_block = 13
    rect_y_mature_block = 1300
    rect_width_mature_block = 195 - rect_x_mature_block
    rect_height_mature_block = 2140 - rect_y_mature_block

    rect_x_expl_block = 13
    rect_y_expl_block  = 930
    rect_width_expl_block  = 195 - rect_x_expl_block
    rect_height_expl_block  = 1300 - rect_y_expl_block

    # Create a scatter plot
    fig, ax = plt.subplots()

    ax.scatter(x_train, y_train, s=0.25, label='Train Traces')
    ax.scatter(x_val, y_val, s=0.25, label='Validation Traces')


    rect_seismic = patches.Rectangle((rect_x_seismic-2, rect_y_seismic-4), rect_width_seismic+4, rect_height_seismic+8, linewidth=2, edgecolor='r', facecolor='none', label='Seismic')
    rect_mature = patches.Rectangle((rect_x_mature_block, rect_y_mature_block), rect_width_mature_block, rect_height_mature_block, linewidth=1, edgecolor='b', facecolor='none', label='Mature Block')
    rect_exploration = patches.Rectangle((rect_x_expl_block, rect_y_expl_block), rect_width_expl_block, rect_height_expl_block, linewidth=1, edgecolor='g', facecolor='none', label='Exploration Block')
    ax.add_patch(rect_seismic)
    ax.add_patch(rect_mature)
    ax.add_patch(rect_exploration)

    # Add explanatory text below the legend
    # Add explanatory text below the legend with specified width
    wrapped_text = "\n".join([f"Total # of traces: {(seis_mature_normalized.shape[0] * seis_mature_normalized.shape[1])} \nExperiment with {porcentaje_entrenamiento}% of traces for training and {porcentaje_validacion}% for validaton "])

    # Adjust layout to make room for the text
    plt.subplots_adjust(right=0.75)
    # Add the wrapped explanatory text to the plot
    fig.text(0.9, 0.4, wrapped_text, ha='center', wrap=True, fontsize=11, color='black')

    # Adjust layout to make room for the text
    plt.subplots_adjust(bottom=0.2)

    # Add labels and title for better understanding
    plt.xlabel('Inline')
    plt.ylabel('Crossline')
    plt.title('Train and Validation Traces')
    plt.legend(bbox_to_anchor=(1, 1.05) )

    # Show the plot
    plt.show()
    
def conv_block(inputs, filters, kernel_size, num_convs=3, activation='leaky_relu'):
    """Create a block of convolutions followed by BatchNorm and activation."""
    x = inputs
    for _ in range(num_convs):
        x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(x)
        x = Activation(activation)(x)
        x = BatchNormalization()(x)
    return x

def downsample_block(inputs):
    """Create a downsampling block with Conv2D and strides."""
    x = MaxPooling2D((2,1))(inputs)
    return x

def upsample_block(inputs, filters, kernel_size, strides=(2,1)):
    """Create an upsampling block with Conv2DTranspose."""
    x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(inputs)
    x = Activation('leaky_relu')(x)
    x = BatchNormalization()(x)
    return x

def unet_model(filtros, given_seed):
    """Build the U-Net model."""
    inputs = Input(shape=(124, 1, 1))

    # First Block
    conv1 = conv_block(inputs, filters=filtros, kernel_size=(5, 1))
    downsampl1 = downsample_block(conv1)

    # Second Block
    conv2 = conv_block(downsampl1, filters=filtros*2, kernel_size=(5, 1))
    downsampl2 = downsample_block(conv2)

    # Third Block
    conv3 = conv_block(downsampl2, filters=filtros*4, kernel_size=(5, 1))

    # First Upsampling
    upsample1 = upsample_block(conv3, filters=filtros*2, kernel_size=(5, 1))

    # Fourth Block
    conv4 = conv_block(upsample1, filters=filtros*2, kernel_size=(5, 1))
    skip = Concatenate()([conv2, conv4])
    
    # Second Upsampling
    upsample2 = upsample_block(skip, filters=filtros, kernel_size=(5, 1))

    # Fifth Block
    conv5 = conv_block(upsample2, filters=filtros, kernel_size=(5, 1))

    # Skip connection
    skip = Concatenate()([conv1, conv5])
    
    #Capa de salida
    outputs = conv_block(skip, filters=1, kernel_size=(5, 1), num_convs=1, activation='tanh')

    #Formateo para ajustar a datos
    outputs = Lambda(lambda x: tf.squeeze(x, axis=[-1, -2]))(outputs)

    model = Model(inputs, outputs)
    return model

def modify_pretrained_model(pretrained_model, filtros, given_seed):
    """    
    Arguments:
    - pretrained_model: The pre-trained model to be modified.
    - filters: Number of filters for the custom layers.
    - seed: Seed for weight initialization of custom layers.
    
    Returns:
    - model: The modified model ready for training.
    """
    # Remove the last two layers (conv10 and Lambda layer)
    model_output = pretrained_model.layers[-3].output  # Output from the layer before `conv10`

    # Add your custom layers
    new_layers = conv_block(model_output, filters=filtros*2, kernel_size=(5, 1))
    new_layers2 = conv_block(new_layers, filters=1, kernel_size=(5, 1), num_convs=1)
    # Add a new Lambda layer for predictions
    new_output = Lambda(lambda x: tf.squeeze(x, axis=[-1, -2]))(new_layers2)

    # Create the new model
    model = Model(inputs=pretrained_model.input, outputs=new_output)

   # Count the number of newly added layers (conv_block layers and Lambda)
    num_new_layers = 0
    for layer in model.layers:
        if layer not in pretrained_model.layers:
            num_new_layers += 1

    # Freeze all the layers except the newly added ones
    for layer in model.layers[:-num_new_layers]:  # Freeze all except the new layers
        layer.trainable = False

    
    return model
    
def simplified_cnn(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder
    # Bloque 1
    x1 = tf.keras.layers.Conv2D(6, (5, 1), strides=1, padding='same', kernel_regularizer=tf.keras.regularizers.l1(0.01))(inputs)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.Conv2D(6, (5, 1), strides=1, padding='same', kernel_regularizer=tf.keras.regularizers.l1(0.01))(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.Conv2D(6, (5, 1), strides=1, padding='same', kernel_regularizer=tf.keras.regularizers.l1(0.01))(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    
    # Bloque 2
    x2 = tf.keras.layers.Conv2D(12, (5, 1), strides=1, padding='same', kernel_regularizer=tf.keras.regularizers.l1(0.01))(x1)
    x2 = tf.keras.layers.LeakyReLU()(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.Conv2D(12, (5, 1), strides=1, padding='same', kernel_regularizer=tf.keras.regularizers.l1(0.01))(x2)
    x2 = tf.keras.layers.LeakyReLU()(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.Conv2D(12, (5, 1), strides=1, padding='same', kernel_regularizer=tf.keras.regularizers.l1(0.01))(x2)
    x2 = tf.keras.layers.LeakyReLU()(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    
    # Output shape: (124, 1, 12)
    drop = tf.keras.layers.Dropout(0.5)(x2)  # 50% of neurons are randomly dropped during training
    
    # flat_bottle_neck = tf.keras.layers.Flatten()(drop)
    # dense_bottle_neck = tf.keras.layers.Dense(1488, activation='leaky_relu')(flat_bottle_neck)
    # reshape_bottleneck = tf.keras.layers.Reshape((124, 1, 12))(dense_bottle_neck)
    
    # Bloque 3
    x3 = tf.keras.layers.Conv2D(24, (5, 1), strides=1, padding='same', kernel_regularizer=tf.keras.regularizers.l1(0.01))(drop)
    x3 = tf.keras.layers.LeakyReLU()(x3)
    x3 = tf.keras.layers.BatchNormalization()(x3)
    x3 = tf.keras.layers.Conv2D(24, (5, 1), strides=1, padding='same', kernel_regularizer=tf.keras.regularizers.l1(0.01))(x3)
    x3 = tf.keras.layers.LeakyReLU()(x3)
    x3 = tf.keras.layers.BatchNormalization()(x3)
    x3 = tf.keras.layers.Conv2D(24, (5, 1), strides=1, padding='same', kernel_regularizer=tf.keras.regularizers.l1(0.01))(x3)
    x3 = tf.keras.layers.LeakyReLU()(x3)
    x3 = tf.keras.layers.BatchNormalization()(x3)
    
        # Bloque 1
    x4 = tf.keras.layers.Conv2D(30, (5, 1), strides=1, padding='same', kernel_regularizer=tf.keras.regularizers.l1(0.01))(x3)
    x4 = tf.keras.layers.LeakyReLU()(x4)
    x4 = tf.keras.layers.BatchNormalization()(x4)
    x4 = tf.keras.layers.Conv2D(30, (5, 1), strides=1, padding='same', kernel_regularizer=tf.keras.regularizers.l1(0.01))(x4)
    x4 = tf.keras.layers.LeakyReLU()(x4)
    x4 = tf.keras.layers.BatchNormalization()(x4)
    x4 = tf.keras.layers.Conv2D(30, (5, 1), strides=1, padding='same', kernel_regularizer=tf.keras.regularizers.l1(0.01))(x4)
    x4 = tf.keras.layers.LeakyReLU()(x4)
    x4 = tf.keras.layers.BatchNormalization()(x4)
    x4 = tf.keras.layers.Conv2D(1, (5, 1), strides=1, padding='same', kernel_regularizer=tf.keras.regularizers.l1(0.01))(x4)
    x4 = tf.keras.layers.LeakyReLU()(x4)
    outputs = tf.keras.layers.BatchNormalization()(x4)

    
    model = tf.keras.Model(inputs, outputs)
    return model

def r2_score(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))  # Residual sum of squares
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))  # Total sum of squares
    return 1 - ss_res / (ss_tot + K.epsilon())  # Para evitar divisiones por cero
