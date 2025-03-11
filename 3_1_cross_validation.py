import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K

from sklearn.metrics import  mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, train_test_split


epocas = 200



seis_full= np.load('data/data_decatur/processed/seismic_full.npy')
phi_full = np.load('data/data_decatur/processed/porosity_full.npy')

well_seismic_data = np.load('data/data_decatur/processed/well_seismic.npy')
well_porosity_data = np.load('data/data_decatur/processed/well_porosity.npy')

phi_full[phi_full<0] = 0


print(phi_full.shape, phi_full.min(), phi_full.max())


depth_start = 5300
depth_end = 7000
depth_step = 20
depth_values = np.arange(depth_start, depth_end , depth_step)
num_ticks = 6  # Adjust the number of ticks as needed
depth_indices = np.linspace(0, len(depth_values) - 1, num_ticks, dtype=int)

inline_number = 83


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


def r2_score(y_true, y_pred):
    """
    Calculate R² score using Keras backend operations.
    
    Parameters:
    -----------
    y_true : tensor
        Ground truth values
    y_pred : tensor
        Predicted values
        
    Returns:
    --------
    tensor
        R² score
    """
    # Convert inputs to tensors if they aren't already
    
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    
    ss_res = K.sum(K.square(y_true - y_pred))  # Residual sum of squares
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))  # Total sum of squares
    return 1 - ss_res / (ss_tot + K.epsilon())  # Para evitar divisiones por cero



class CustomSeismicLoss(tf.keras.losses.Loss):
    def __init__(self,
                 model,  # The neural network model
                 well_seismic_data,  # Seismic data at well locations S_w
                 well_porosity_data,  # Observed porosity at well locations φ_w
                 lambda_param=0.5,
                 name='custom_seismic_loss'):
        """
        Custom loss function for seismic to porosity mapping.
        
        Args:
            model (tf.keras.Model): Neural network model to predict porosity
            well_seismic_data (tf.Tensor): Seismic data at well locations S_w
            well_porosity_data (tf.Tensor): Observed porosity at well locations φ_w
            lambda_param (float): Regularization parameter
            name (str): Name of the loss function
        """
        super().__init__(name=name)
        self.model = model
        self.well_seismic_data = tf.convert_to_tensor(well_seismic_data, dtype=tf.float32)
        self.well_porosity_data = tf.convert_to_tensor(well_porosity_data, dtype=tf.float32)
        self.lambda_param = lambda_param
    
    def call(self, y_true, y_pred):
        """
        Compute the custom loss.
        
        Args:
            y_true (tf.Tensor): Actual porosity values
            y_pred (tf.Tensor): Predicted porosity values
        
        Returns:
            tf.Tensor: Computed loss value
        """
        # First term: L2 norm of prediction error for full dataset
        first_term = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Second term: Predict porosity using well seismic data and compare with well porosity
        y_pred_well = self.model(self.well_seismic_data, training=False)
        second_term = (self.lambda_param / 2) * tf.reduce_mean(tf.square(self.well_porosity_data - y_pred_well ))
        
        # Total loss is the sum of both terms
        total_loss = first_term + second_term
        
        return total_loss


class LearningRateSchedulerOnR2(tf.keras.callbacks.Callback):
    def __init__(self, patience=5, factor=0.5, min_lr=1e-6):
        super().__init__()
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.r2_scores = []
        self.consecutive_decrease_count = 0

    def on_epoch_end(self, epoch, logs=None):
        r2_score = logs.get('val_r2_score')
        if r2_score is None:
            print("R2 score is not logged; skipping learning rate adjustment.")
            return
        
        self.r2_scores.append(r2_score)
        
        if len(self.r2_scores) > 1:
            if r2_score < self.r2_scores[-2]:
                self.consecutive_decrease_count += 1
            else:
                self.consecutive_decrease_count = 0
            
        if self.consecutive_decrease_count >= self.patience:
            try:
                # Retrieve the old learning rate
                old_lr = float(self.model.optimizer.learning_rate.numpy())
                print(f"Old learning rate: {old_lr:.6f}")

                # Compute the new learning rate
                new_lr = max(old_lr * self.factor, self.min_lr)
                print(f"Computed new learning rate: {new_lr}, type: {type(new_lr)}")

                # Safely update the learning rate
                self.model.optimizer.learning_rate.assign(new_lr)
                print(f"Learning rate successfully updated to: {new_lr:.6f}")

                self.consecutive_decrease_count = 0  # Reset counter
            except Exception as e:
                print(f"Error during learning rate adjustment: {e}")
                print(f"Learning rate object: {self.model.optimizer.learning_rate}, type: {type(self.model.optimizer.learning_rate)}")
                raise


        
        current_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        print(f"Epoch {epoch+1}: R2 score: {r2_score:.4f}, Learning rate: {current_lr:.6f}")




class LearningRateLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.learning_rates = []

    def on_epoch_end(self, epoch, logs=None):
        # Retrieve the current learning rate and store it
        current_lr = float(self.model.optimizer.learning_rate.numpy())
        self.learning_rates.append(current_lr)
        print(f"Learning rate for epoch {epoch+1}: {current_lr:.6f}")



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
model_own = simplified_cnn(input_shape)
print(model_own.summary())


# Shuffle and Split Full Dataset into Train/Test
test_size = 0.2  # 20% for the test set
pos = np.random.permutation(Y_norm.shape[0])
X_shuffled, Y_shuffled = X_norm[pos], Y_norm[pos]

X_train, X_test, Y_train, Y_test = train_test_split(X_shuffled, Y_shuffled, test_size=test_size)

# K-Fold Cross-Validation on the Training Data
k = 5  # Number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)

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
        lambda_param=0.1
    )

    initial_lr = 0.001
    model_own.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr), 
                      loss=custom_loss, 
                      metrics=['mae', r2_score])  
    
    # Define callbacks for this fold
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'models/fold_{fold+1}_training_paul.weights.h5',
        save_weights_only=True,
        verbose=1
    )

    lr_scheduler = LearningRateSchedulerOnR2(patience=1, factor=0.8, min_lr=1e-6)
    
    callback_early = tf.keras.callbacks.EarlyStopping(monitor='val_r2_score', 
                                                      patience=10, 
                                                      mode='max',
                                                      min_delta=0.1)
    lr_logger = LearningRateLogger()

    # Train the model
    history = model_own.fit(
        X_train_fold, Y_train_fold,
        validation_data=(X_val_fold, Y_val_fold),
        epochs=epocas, 
        batch_size=512, 
        shuffle=True,
        callbacks=[checkpoint, lr_logger, lr_scheduler]
    )

    # Evaluate the model on the validation set
    val_predictions = model_own.predict(X_val_fold)
    fold_r2 = r2_score(Y_val_fold, val_predictions)
    cv_scores.append(fold_r2)
    histories.append(history)
    print(f"R2 Score for Fold {fold+1}: {fold_r2:.4f}")

# Evaluate on the Test Set (Unseen Data)
test_predictions = model_own.predict(X_test)
test_r2 = r2_score(Y_test, test_predictions)

# Aggregate Cross-Validation Results
print("\nCross-Validation Results:")
print(f"R2 Scores: {cv_scores}")
print(f"Mean R2 Score: {np.mean(cv_scores):.4f}")
print(f"Standard Deviation: {np.std(cv_scores):.4f}")
print(f"\nR2 Score on Test Set: {test_r2:.4f}")

print(cv_scores)

epochs = [i for i in range(len(history.history['r2_score']))]

fig, ax = plt.subplots(5,3)
fig.set_size_inches(16,30)
counter = 0
for history in histories:
    
    train_acc = history.history['r2_score']
    train_loss = history.history['loss']
    val_acc = history.history['val_r2_score']
    val_loss = history.history['val_loss']
    #train_lr = history_own.history['learning_rate']

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


    ax[counter, 2].plot(epochs, lr_logger.learning_rates, 'go-', label='lr-train')
    ax[counter, 2].set_title('Learning Rate train')
    ax[counter, 2].legend()
    ax[counter, 2].set_xlabel('epochs')
    ax[counter,2].set_ylabel('Learning Rate')
    counter += 1
fig.savefig("./plots/histories.png", format="png", bbox_inches="tight")
plt.show()

print(history.history.keys())

loss, mae, accuracy = model_own.evaluate(X_test, Y_test)

print(f'loss: {loss}')
print(f'mae: {mae}')
print(f'r2: {accuracy}')

y_pred = model_own.predict(X_test)

Y_test = np.squeeze(Y_test)
y_pred = np.squeeze(y_pred)

print(Y_test.shape, y_pred.shape)

rmse_own_cnn = np.sqrt(mean_squared_error(Y_test, y_pred))
MAE_own_cnn = mean_absolute_error(Y_test, y_pred)
r2_own_cnn = r2_score(Y_test, y_pred)

print("RMSE : % f" %(rmse_own_cnn))
print("MAE : % f" %(MAE_own_cnn))
print("R2 : % f" %(r2_own_cnn))

y_pred_unscaled = unscale_from_range(y_pred, original_min= phi_full.min(), original_max=phi_full.max())
y_test_unscaled = unscale_from_range(Y_test, original_min= phi_full.min(), original_max=phi_full.max())

print(y_pred_unscaled.shape)
print(y_test_unscaled.shape)