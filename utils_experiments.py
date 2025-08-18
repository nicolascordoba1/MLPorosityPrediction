import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np 
import tensorflow as tf
from keras.layers import Activation, Conv2D, Conv2DTranspose, Lambda, Input, Concatenate, BatchNormalization, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


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

class R2Nicolas(tf.keras.metrics.Metric):
    def __init__(self, name='r2_score', **kwargs):
        # Initialize the parent class
        super(R2Nicolas, self).__init__(name=name, **kwargs)
        
        # Create variables to accumulate values during training
        self.ss_res = self.add_weight(name='ss_res', initializer='zeros')
        self.ss_tot = self.add_weight(name='ss_tot', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Flatten the predictions and true values
        y_pred = tf.reshape(y_pred, [-1])
        y_true = tf.reshape(y_true, [-1])
        
        # Calculate mean of true values
        y_mean = tf.reduce_mean(y_true)
        
        # Calculate sum of squared residuals
        residuals = tf.square(y_true - y_pred)
        self.ss_res.assign_add(tf.reduce_sum(residuals))
        
        # Calculate total sum of squares
        total = tf.square(y_true - y_mean)
        self.ss_tot.assign_add(tf.reduce_sum(total))

    def result(self):
        # Calculate and return R² score
        return 1 - self.ss_res / (self.ss_tot + K.epsilon())

    def reset_state(self):
        # Reset accumulated values
        self.ss_res.assign(0.0)
        self.ss_tot.assign(0.0)

class CustomSeismicLoss(tf.keras.losses.Loss):
    def __init__(self,
                 model,
                 well_seismic_data,
                 well_porosity_data,
                 lambda_param=0.5,
                 name='custom_seismic_loss'):
        super().__init__(name=name)
        self.model = model
        self.lambda_param = lambda_param
        self.well_seismic_data = tf.convert_to_tensor(well_seismic_data, dtype=tf.float32)
        self.well_porosity_data = tf.convert_to_tensor(well_porosity_data, dtype=tf.float32)

        # Loss checkpoint
        self.last_first_term = tf.Variable(0.0)
        self.last_second_term = tf.Variable(0.0)

    def calculate_mse(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_pred - y_true))
        self.last_first_term.assign(mse)
        return mse

    def calculate_well_term(self):
        if self.lambda_param == 0:
            self.last_second_term.assign(0.0)
            return 0.0

        y_pred_well = self.model(self.well_seismic_data, training=True)
        well_term = (self.lambda_param / 2) * tf.reduce_mean(tf.square(y_pred_well - self.well_porosity_data))
        self.last_second_term.assign(well_term)
        return well_term

    def call(self, y_true, y_pred):
        return self.calculate_mse(y_true, y_pred) + self.calculate_well_term()

class LossComponentCallback(tf.keras.callbacks.Callback):
    def __init__(self, custom_loss):
        super().__init__()
        self.custom_loss = custom_loss
        # Initialize lists to store loss components
        self.first_terms = []
        self.second_terms = []
        self.total_losses = []
        self.val_total_losses = []

    def on_epoch_end(self, epoch, logs={}):
        first_term = float(tf.keras.backend.get_value(self.custom_loss.last_first_term))

        # Only get second term if lambda is not 0
        if self.custom_loss.lambda_param > 0:
            second_term = float(tf.keras.backend.get_value(self.custom_loss.last_second_term))
        else:
            second_term = 0.0

        total_loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)

        # Store the values
        self.first_terms.append(first_term)
        self.second_terms.append(second_term)
        self.total_losses.append(total_loss)
        self.val_total_losses.append(val_loss)

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch + 1} Loss Components:")
            print(f"First Term (MSE): {first_term:.6f}")
            print(f"Second Term (Well Constraint): {second_term:.6f}")
            print(f"Total Loss: {total_loss:.6f}")
            print(f"Validation Loss: {val_loss:.6f}")

    def plot_losses(self):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))
        epochs = range(1, len(self.first_terms) + 1)

        # Plot all components
        plt.plot(epochs, self.first_terms, 'b-', label='First Term (MSE)')
        plt.plot(epochs, self.second_terms, 'g-', label='Second Term (Well Constraint)')
        plt.plot(epochs, self.total_losses, 'r-', label='Total Training Loss')
        plt.plot(epochs, self.val_total_losses, 'r--', label='Total Validation Loss')

        plt.title('Loss Components Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Add log scale option if the losses vary by orders of magnitude
        plt.yscale('log')

        plt.tight_layout()
        plt.show()
        
def simplified_cnn(input_shape):

    inputs = tf.keras.Input(shape=input_shape)

    # Bloque 1
    x1 = tf.keras.layers.Conv2D(8, (15, 1), strides=1, padding='same')(inputs)
    x1 = tf.keras.layers.Activation('leaky_relu')(x1)

    # Bloque 2
    x2 = tf.keras.layers.Conv2D(16, (15, 1), strides=1, padding='same')(x1)
    x2 = tf.keras.layers.Activation('leaky_relu')(x2)
    
    conv_bottleneck = tf.keras.layers.Conv2D(4, (15, 1), strides=1, padding='same')(x2)
    conv_bottleneck = tf.keras.layers.Activation('leaky_relu')(conv_bottleneck)
    
    drop2 = tf.keras.layers.Dropout(0.1)(conv_bottleneck)
    
    flat_bottle_neck = tf.keras.layers.Flatten()(drop2)
    dense_bottle_neck = tf.keras.layers.Dense(input_shape[0]*4, activation='leaky_relu')(flat_bottle_neck)
    reshape_bottleneck = tf.keras.layers.Reshape((input_shape[0], 1, 4))(dense_bottle_neck)

    # Bloque 3
    x3 = tf.keras.layers.Conv2D(32, (15, 1), strides=1, padding='same')(reshape_bottleneck)
    x3 = tf.keras.layers.Activation('leaky_relu')(x3)

    # Bloque 4
    x4 = tf.keras.layers.Conv2D(64, (15, 1), strides=1, padding='same')(x3)
    x4 = tf.keras.layers.Activation('leaky_relu')(x4)
    
    
    x5 = tf.keras.layers.Conv2D(1, (15, 1), strides=1, padding='same')(x4)
    outputs = tf.keras.layers.Activation('tanh')(x5)

    model = tf.keras.Model(inputs, outputs)
    return model