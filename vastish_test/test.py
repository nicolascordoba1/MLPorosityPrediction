import os

import matplotlib.pyplot as plt
import matplotlib

import numpy as np

import tensorflow as tf
from keras.layers import Activation, BatchNormalization, Conv2D, Conv1D
from keras import initializers

from sklearn import model_selection

import lasio
matplotlib.use('TkAgg')
epocas=100

def visualizacion_resultados(history):
    epochs = [i for i in range(epocas)]

    train_acc = history.history['r2_nicolas']
    train_loss = history.history['loss']
    train_lr = history.history['lr']
    val_acc = history.history['val_r2_nicolas']
    val_loss = history.history['val_loss']
    

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
    
    fig.savefig('./plots/training.png')

    plt.show()
# Load data
seis_mature= np.load('../data_decatur/processed/mature_block.npy')
phi_mature = np.load('../data_decatur/processed/mature_block_porosity.npy')

seis_exploration = np.load('../data_decatur/processed/exploration_block.npy')
ccs1 = lasio.read('../data_decatur/wells/ccs1_phie_smoothed.las')
vw1 = lasio.read('../data_decatur/wells/vw1_phie_smoothed.las')
seis_mature.shape
# Shape and Statistics
fig, ax = plt.subplots(1,2, figsize = (10, 4))
fig.suptitle('Random inline')
im1 = ax[0].imshow(seis_mature[100,:,:].T, cmap='gray_r')
ax[0].set_title('Sísmica')
ax[0].set_aspect('auto')
fig.colorbar(im1, ax=ax[0], shrink=1)

im2 = ax[1].imshow(phi_mature[100,:,:].T, cmap='jet')
ax[1].set_title('Porosidad')
ax[1].set_aspect('auto')
fig.colorbar(im2, ax=ax[1], shrink=1)

fig.tight_layout()
fig.savefig("./plots/section_original.png")
plt.show()
depth = np.arange(4530, 7010, 20)
fig, ax = plt.subplots(1,2, figsize = (4, 10))
fig.suptitle('Inline CCS1 VW1')
ax[0].plot(seis_mature[15,15,:], depth)
ax[0].set_title('Seismic')
ax[0].set_aspect('auto')
ax[1].plot(phi_mature[15,15,:], depth)
ax[1].set_title('Porosity')
ax[1].set_aspect('auto')
fig.tight_layout()
fig.savefig("./plots/traza_original.svg")
plt.show()
seis_mature.shape
phi_mature.shape
phi_mature.mean()
phi_mature.min()
phi_mature.max()
plt.hist(phi_mature.ravel())
# Normalizacion de Datos
def min_max_scale(x, min, max):

  x_std = (x - min) / (max - min)
  x_scaled = x_std * 2 - 1
  return x_scaled

def inverse_min_max_scale(x, min, max):

  x_normalized = (x + 1) / 2
  x_unscaled = x_normalized * (max - min) + min
  return x_unscaled
phi_mature[phi_mature<0] = 0.0001
phi_max=np.max(phi_mature) #can also take 1 or critical porosity (0.4)
phi_min=np.min(phi_mature) #can also take 0
phi_scaled = min_max_scale(phi_mature, min= phi_min, max=phi_max)
phi_unscaled = inverse_min_max_scale(phi_scaled, min= phi_min, max=phi_max)
plt.hist(phi_scaled.ravel())
plt.hist(phi_unscaled.ravel())
# Normalizing the input and output data  

seis_normalized = (seis_mature - np.min(seis_mature))/(np.max(seis_mature)-np.min(seis_mature))


#true_output=np.zeros((seis_normalized.shape[0],seis_normalized.shape[1]))
#true_output[:,0:199]=(phi_mature-phi_min)/(phi_max-phi_min)
np.min(seis_normalized), np.max(seis_normalized)
np.min(phi_scaled), np.max(phi_scaled)
seis_mature.shape, phi_scaled.shape
# Preparación de Datos

train_wells = int((seis_normalized.shape[0] * seis_normalized.shape[1]) * 0.01)
val_wells = 10


# Create a mask to keep track of which indices have been used
mask = np.ones(seis_normalized.shape[:2], dtype=bool)

def extract_traces(numer_of_wells):
    x_seismic = []
    y_porosity = []
    for _ in range(numer_of_wells):
        random_il = np.random.randint(0, seis_normalized.shape[0])
        random_xl = np.random.randint(0, seis_normalized.shape[1])
        
        # Ensure we pick an unused index
        while not mask[random_il, random_xl]:
            random_il = np.random.randint(0, seis_normalized.shape[0])
            random_xl = np.random.randint(0, seis_normalized.shape[1])
        
        X_chosen = seis_normalized[random_il, random_xl, :]
        Y_chosen = phi_scaled[random_il, random_xl, :]
        
        x_seismic.append(np.expand_dims(X_chosen, axis=0))
        y_porosity.append(np.expand_dims(Y_chosen, axis=0))
        
        # Mark this index as used
        mask[random_il, random_xl] = False

    x_seismic = np.concatenate(x_seismic, axis=0)
    y_porosity = np.concatenate(y_porosity, axis=0)
    
    return x_seismic, y_porosity
    
X_train, Y_train = extract_traces(train_wells)
X_val, Y_val = extract_traces(val_wells)
    
# Remaining data for test
X_test = seis_normalized[mask]
Y_test = phi_scaled[mask]

print("Number of traces:", seis_normalized.shape[0] * seis_normalized.shape[1])
print("Training set shapes:", X_train.shape, Y_train.shape)
print("Validation set shapes:", X_val.shape, Y_val.shape)
print("Test set shapes:", X_test.shape, Y_test.shape)


print('acabé')