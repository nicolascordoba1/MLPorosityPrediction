from utils_experiments import R2Nicolas, scale_to_range, unscale_from_range, simplified_cnn, CustomSeismicLoss, LossComponentCallback

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Activation, Dense, Flatten
from keras import initializers, regularizers

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, train_test_split

import pandas as pd
import time

import matplotlib.pyplot as plt

epocas = 50

seis_full= np.load('data/data_decatur/processed/seismic_mature_block.npy')
phi_full = np.load('data/data_decatur/processed/porosity_mature_block.npy')

well_seismic_data = np.load('data/data_decatur/processed/well_seismic.npy')
well_porosity_data = np.load('data/data_decatur/processed/well_porosity.npy')

phi_full[phi_full<0] = 0


print(phi_full.shape, phi_full.min(), phi_full.max())


X = seis_full.reshape(-1,86,1,1)
Y = phi_full.reshape(-1,86,1,1)


X_norm = scale_to_range(X)
Y_norm = scale_to_range(Y)
well_seismic_data = scale_to_range(well_seismic_data)
well_porosity_data = scale_to_range(well_porosity_data)

print(X_norm.shape, X_norm.min(), X_norm.max())
print(Y_norm.shape, Y_norm.min(), Y_norm.max())
print(well_seismic_data.shape, well_seismic_data.min(), well_seismic_data.max())
print(well_porosity_data.shape, well_porosity_data.min(), well_porosity_data.max())

# Definir las dimensiones de entrada
input_shape = (86, 1, 1)
model_own = simplified_cnn(input_shape)
print(model_own.summary())


# Shuffle and Split Full Dataset into Train/Test
test_size = 0.3  # 30% for the test set
pos = np.random.permutation(Y_norm.shape[0])
X_shuffled, Y_shuffled = X_norm[pos], Y_norm[pos]

X_train, X_test, Y_train, Y_test = train_test_split(X_shuffled, Y_shuffled, test_size=test_size)

# K-Fold Cross-Validation on the Training Data
k = 5  # Number of folds
kf = KFold(n_splits=k, shuffle=True)

cv_scores = []
histories = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"Fold {fold+1}/{k}")
    
    # Split training data into K-Fold subsets
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    Y_train_fold, Y_val_fold = Y_train[train_idx], Y_train[val_idx]
    
    # Create a new model instance for this fold
    model_own = simplified_cnn(input_shape)
    
    # Compile the model
    custom_loss = CustomSeismicLoss(
        model_own,
        well_seismic_data, 
        well_porosity_data, 
        lambda_param=10
    )

    initial_lr = 0.001
    model_own.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr), 
                      loss=custom_loss, 
                      metrics=[R2Nicolas()])  
    
    # Define callbacks for this fold
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'models/fold_{fold+1}_proposed.weights.h5',
        save_weights_only=True,
        verbose=1
    )

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_r2_score',
        mode='max',  # since we want to maximize R2
        factor=0.1,
        patience=1,
        min_lr=1e-6,
        verbose=1  # This will print when learning rate changes
    )
    
    loss_callback = LossComponentCallback(custom_loss)
    
    callback_early = tf.keras.callbacks.EarlyStopping(monitor='val_r2_score', 
                                                      patience=10, 
                                                      mode='max',
                                                      min_delta=0.1)

    # Train the model
    history = model_own.fit(
        X_train_fold, Y_train_fold,
        validation_data=(X_val_fold, Y_val_fold),
        epochs=epocas, 
        batch_size=512, 
        shuffle=True,
        callbacks=[checkpoint, loss_callback, lr_scheduler]
    )

    # Evaluate the model on the validation set
    test_predictions_innie = model_own.predict(X_test)
    fold_r2 = r2_score(np.ravel(Y_test), np.ravel(test_predictions_innie))
    cv_scores.append(fold_r2)
    histories.append(history)
    print(f"R2 Score for Fold {fold+1}: {fold_r2:.4f}")

# Aggregate Cross-Validation Results
print("\nCross-Validation Results:")
print(f"R2 Scores: {cv_scores}")
print(f"Mean R2 Score: {np.mean(cv_scores):.4f}")
print(f"Standard Deviation: {np.std(cv_scores):.4f}")
#print(f"\nR2 Score on Test Set: {test_r2:.4f}")

#print(cv_scores)

epochs = [i for i in range(len(history.history['r2_score']))]

fig, ax = plt.subplots(5,3)
fig.set_size_inches(16,30)
counter = 0
print(history.history.keys())
for history in histories:
    
    train_acc = history.history['r2_score']
    train_loss = history.history['loss']
    val_acc = history.history['val_r2_score']
    val_loss = history.history['val_loss']
    train_lr = history.history['learning_rate']

    ax[counter,0].plot(epochs, train_acc, 'go-', label='accuracy-train')
    ax[counter,0].plot(epochs, val_acc, 'ro-', label='accuracy-val')
    ax[counter,0].set_title('Accuracy train')
    ax[counter,0].legend()
    ax[counter,0].set_xlabel('epochs')
    ax[counter,0].set_ylabel('accuracy')
    ax[counter,0].set_ylim(0,1)

    ax[counter,1].plot(epochs, train_loss, 'go-', label='loss-train')
    ax[counter,1].plot(epochs, val_loss, 'ro-', label='loss-val')
    ax[counter,1].set_title('Loss train')
    ax[counter,1].legend()
    ax[counter,1].set_xlabel('epochs')
    ax[counter,1].set_ylabel('loss')
    ax[counter,1].set_ylim(0,2)


    ax[counter, 2].plot(epochs, train_lr, 'go-', label='lr-train')
    ax[counter, 2].set_title('Learning Rate train')
    ax[counter, 2].legend()
    ax[counter, 2].set_xlabel('epochs')
    ax[counter,2].set_ylabel('Learning Rate')
    counter += 1
fig.savefig("./plots/histories.png", format="png", bbox_inches="tight")
plt.show()

print(history.history.keys())
