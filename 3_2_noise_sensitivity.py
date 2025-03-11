import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import r2_score
from utils_experiments import scale_to_range, unscale_from_range, simplified_cnn

depth_start = 4700
depth_end = 6410
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
 
noise_prcentage = np.array([ 0.10,  0.20,  0.30, 0.40])

# Results dictionary
results = {"Noise Level": [], "SNR": [], "SSIM": [], "R2": [], "Max Difference": []}


# Load data
seis_full = np.load('data/data_decatur/processed/seismic_exploration_block.npy')
phi_full = np.load('data/data_decatur/processed/porosity_exploration_block.npy')
phi_full[phi_full < 0] = 0

# Test inline

max_amplitud = abs(seis_full).max()
noise_levels = noise_prcentage * max_amplitud
# Model setup
input_shape = (86, 1, 1)
model = simplified_cnn(input_shape)
print(model.summary())

model.load_weights('models/proposed.weights.h5')

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
    snr_value = calculate_snr(np.ravel(phi_full), np.ravel(y_pred_unscaled_noisy))
    ssim_value = ssim(np.ravel(phi_full), np.ravel(y_pred_unscaled_noisy), data_range=0.3)
    r2_value = r2_score(np.ravel(phi_full), np.ravel(y_pred_unscaled_noisy))

    # Append results
    results["SNR"].append(snr_value)
    results["SSIM"].append(ssim_value)
    results["R2"].append(r2_value)
    
    phi_difference_noisy = y_pred_unscaled_noisy - phi_full
    # Calculate absolute difference and mean
    #mean_difference = np.mean(phi_difference_noisy)
    max_absolute_difference = np.max(phi_difference_noisy)

    # Store in results
    results["Max Difference"].append(max_absolute_difference)
    #Plot
    
    phi_difference_noisy = y_pred_unscaled_noisy - phi_full
    
    inline = 83-40
    fig, ax = plt.subplots(3, 1, figsize=(7.5, 15))

    fig.suptitle('Noise Addition Test with Noise Level: ' + str(noise_prcentage[idx]*100), fontsize=12)  # Adjust figure title size

    im1 = ax[0].imshow(noisy_seismic[inline,:,:].T, cmap='Greys')
    ax[0].set_title('Seismic', fontsize=12)  # Adjust title size
    ax[0].set_aspect('auto')
    ax[0].set_yticks(depth_indices)
    ax[0].set_yticklabels(depth_values[depth_indices], fontsize=12)  # Adjust y-tick label size
    ax[0].set_xlabel('Crossline', fontsize=12)  # Adjust x-label size
    ax[0].set_ylabel('Depth', fontsize=12)  # Adjust y-label size
    fig.colorbar(im1, ax=ax[0], shrink=1)

    im2 = ax[1].imshow(y_pred_unscaled_noisy[inline,:,:].T, vmin=0, vmax=0.3, cmap='jet')
    ax[1].set_title('Estimated Porosity', fontsize=12)
    ax[1].set_aspect('auto')
    ax[1].set_yticks(depth_indices)
    ax[1].set_yticklabels(depth_values[depth_indices], fontsize=12)
    ax[1].set_xlabel('Crossline', fontsize=12)
    fig.colorbar(im2, ax=ax[1], shrink=1)

    im3 = ax[2].imshow(phi_difference_noisy[inline,:,:].T, vmin=-0.1, vmax=0.1,cmap='jet')
    ax[2].set_title('Difference', fontsize=12)
    ax[2].set_aspect('auto')
    ax[2].set_yticks(depth_indices)
    ax[2].set_yticklabels(depth_values[depth_indices], fontsize=12)
    ax[2].set_xlabel('Crossline', fontsize=12)
    fig.colorbar(im3, ax=ax[2], shrink=1)

    #fig.tight_layout()
    fig.savefig(f"./plots/{noise_prcentage[idx]*100}_noise_test.pdf", format="pdf",  dpi=300)

print(results)
results_dataframe = pd.DataFrame(results)
print(results_dataframe)
latex_table = results_dataframe.to_latex(index=False, float_format="%.3f")
print(latex_table)