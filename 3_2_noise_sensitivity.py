import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import r2_score
from utils_experiments import scale_to_range, unscale_from_range

depth_start = 4700
depth_end = 6400
depth_step = 20
depth_values = np.arange(depth_start, depth_end , depth_step)
num_ticks = 6  # Adjust the number of ticks as needed
depth_indices = np.linspace(0, len(depth_values) - 1, num_ticks, dtype=int)

# Function to calculate SNR
def calculate_snr(true, predicted):
    signal_power = np.mean(true ** 2)
    noise_power = np.mean((true - predicted) ** 2)
    return 10 * np.log10(signal_power / noise_power)

# Noise levels to test
 
noise_prcentage = np.array([0, 0.05, 0.10, 0.15, 0.20, 0.30])

# Results dictionary
results = {"Noise Level": [], "SNR": [], "SSIM": [], "R2": [], "Absolute Difference": []}


# Load data
seis_full = np.load('data/data_decatur/processed/seismic_exploration_block.npy')
phi_full = np.load('data/data_decatur/processed/porosity_exploration_block.npy')
phi_full[phi_full < 0] = 0

# Test inline

max_amplitud = abs(seis_full).max()
noise_levels = noise_prcentage * max_amplitud
# Model setup
def simplified_cnn(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder
    # Bloque 1
    x1 = tf.keras.layers.Conv2D(6, (5, 1), strides=1, padding='same', kernel_regularizer=tf.keras.regularizers.l1(0.01))(inputs)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)

    
    # Bloque 2
    x2 = tf.keras.layers.Conv2D(12, (5, 1), strides=1, padding='same', kernel_regularizer=tf.keras.regularizers.l1(0.01))(x1)
    x2 = tf.keras.layers.LeakyReLU()(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    
    # Output shape: (86, 1, 12)
    drop = tf.keras.layers.Dropout(0.5)(x2)  # 50% of neurons are randomly dropped during training
    
    flat_bottle_neck = tf.keras.layers.Flatten()(drop)
    dense_bottle_neck = tf.keras.layers.Dense(1032, activation='leaky_relu')(flat_bottle_neck)
    reshape_bottleneck = tf.keras.layers.Reshape((86, 1, 12))(dense_bottle_neck)
    
    # Bloque 3
    x3 = tf.keras.layers.Conv2D(24, (5, 1), strides=1, padding='same', kernel_regularizer=tf.keras.regularizers.l1(0.01))(reshape_bottleneck)
    x3 = tf.keras.layers.LeakyReLU()(x3)
    x3 = tf.keras.layers.BatchNormalization()(x3)

    
    # Bloque 4
    x4 = tf.keras.layers.Conv2D(30, (5, 1), strides=1, padding='same', kernel_regularizer=tf.keras.regularizers.l1(0.01))(x3)
    x4 = tf.keras.layers.LeakyReLU()(x4)
    x4 = tf.keras.layers.BatchNormalization()(x4)

    
    x4 = tf.keras.layers.Conv2D(1, (5, 1), strides=1, padding='same', kernel_regularizer=tf.keras.regularizers.l1(0.01))(x4)
    x4 = tf.keras.layers.LeakyReLU()(x4)
    outputs = tf.keras.layers.BatchNormalization()(x4)

    
    model = tf.keras.Model(inputs, outputs)
    return model

# Definir las dimensiones de entrada
input_shape = (86, 1, 1)
model = simplified_cnn(input_shape)
print(model.summary())
model.load_weights('models/training_NFE.weights.h5')

for idx, noise_level in enumerate(noise_levels):
    results["Noise Level"].append(noise_prcentage[idx]*100)
    # Generate noise and apply it
    noise = np.random.normal(loc=0, scale=noise_level, size=seis_full.shape)
    noisy_seismic = seis_full + noise

    # Preprocess inputs
    X_noisy = noisy_seismic.reshape(-1, 86, 1, 1)
    X_noisy_norm = scale_to_range(X_noisy)

    # Predictions
    y_pred_noisy = model.predict(X_noisy_norm)

    # Rescale outputs
    y_pred_unscaled_noisy = unscale_from_range(
        y_pred_noisy, original_min=phi_full.min(), original_max=phi_full.max()
    ).reshape(143, 370, 86)
    
    # Calculate metrics
    snr_value = calculate_snr(phi_full, y_pred_unscaled_noisy)
    ssim_value = ssim(phi_full, y_pred_unscaled_noisy, data_range=0.3)
    r2_value = r2_score(phi_full.flatten(), y_pred_unscaled_noisy.flatten())

    # Append results
    results["SNR"].append(snr_value)
    results["SSIM"].append(ssim_value)
    results["R2"].append(r2_value)
    
    phi_difference_noisy = y_pred_unscaled_noisy - phi_full
    # Calculate absolute difference and mean
    absolute_difference = np.abs(phi_difference_noisy)
    max_absolute_difference = np.max(absolute_difference)

    # Store in results
    results["Absolute Difference"].append(max_absolute_difference)
    #Plot
    
    phi_difference_noisy = y_pred_unscaled_noisy - phi_full
    
    inline = 83-40
    fig, ax = plt.subplots(1, 4, figsize=(25, 5))

    fig.suptitle('Noise Addition Test with Noise Level: ' + str(noise_level), fontsize=40)  # Adjust figure title size

    im1 = ax[0].imshow(noisy_seismic[inline,:,:].T, cmap='seismic')
    ax[0].set_title('Seismic', fontsize=30)  # Adjust title size
    ax[0].set_aspect('auto')
    ax[0].set_yticks(depth_indices)
    ax[0].set_yticklabels(depth_values[depth_indices], fontsize=12)  # Adjust y-tick label size
    ax[0].set_xlabel('Crossline', fontsize=18)  # Adjust x-label size
    ax[0].set_ylabel('Depth', fontsize=18)  # Adjust y-label size
    fig.colorbar(im1, ax=ax[0], shrink=1)

    im2 = ax[1].imshow(y_pred_unscaled_noisy[inline,:,:].T, vmin=0, vmax=0.3, cmap='jet')
    ax[1].set_title('Estimated Porosity', fontsize=30)
    ax[1].set_aspect('auto')
    ax[1].set_yticks(depth_indices)
    ax[1].set_yticklabels(depth_values[depth_indices], fontsize=18)
    ax[1].set_xlabel('Crossline', fontsize=18)
    fig.colorbar(im2, ax=ax[1], shrink=1)

    im3 = ax[2].imshow(phi_full[inline,:,:].T, vmin=0, vmax=0.3, cmap='jet')
    ax[2].set_title('Ground Truth Porosity', fontsize=30)
    ax[2].set_aspect('auto')
    ax[2].set_yticks(depth_indices)
    ax[2].set_yticklabels(depth_values[depth_indices], fontsize=18)
    ax[2].set_xlabel('Crossline', fontsize=18)
    fig.colorbar(im3, ax=ax[2], shrink=1)

    im4 = ax[3].imshow(phi_difference_noisy[inline,:,:].T, cmap='jet')
    ax[3].set_title('Difference', fontsize=30)
    ax[3].set_aspect('auto')
    ax[3].set_yticks(depth_indices)
    ax[3].set_yticklabels(depth_values[depth_indices], fontsize=18)
    ax[3].set_xlabel('Crossline', fontsize=18)
    fig.colorbar(im4, ax=ax[3], shrink=1)

    fig.tight_layout()
    fig.savefig(f"./plots/{noise_level}_noise_test.png", format="png", bbox_inches="tight")

print(results)
results_dataframe = pd.DataFrame(results)
print(results_dataframe)
latex_table = results_dataframe.to_latex(index=False, float_format="%.3f")
print(latex_table)