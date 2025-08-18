
# # Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.layers import Activation, Conv1D, Flatten, Dense
from keras import initializers
from tensorflow.keras.metrics import R2Score

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import time

import gc

from utils_experiments import simplified_cnn, CustomSeismicLoss, R2Nicolas, LossComponentCallback, scale_to_range, unscale_from_range



# Set GPU memory growth to prevent pre-allocating large chunks of memory
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
# # Load data

seis_mature= np.load('../2024_tesis_maestria/data/data_decatur/processed/seismic_mature_block.npy')
phi_mature = np.load('../2024_tesis_maestria/data/data_decatur/processed/porosity_mature_block.npy')

seis_exploration = np.load('../2024_tesis_maestria/data/data_decatur/processed/seismic_exploration_block.npy')
phi_exploration = np.load('../2024_tesis_maestria/data/data_decatur/processed/porosity_exploration_block.npy')

well_seismic_data = np.load('../2024_tesis_maestria/data/data_decatur/processed/well_seismic.npy')
well_porosity_data = np.load('../2024_tesis_maestria/data/data_decatur/processed/well_porosity.npy')


epocas=50

# # Graficos

def visualizacion_resultados(history, epocas, percent):
    epochs = [i for i in range(epocas)]

    train_acc = history.history['r2_score']
    train_loss = history.history['loss']
    train_lr = history.history['learning_rate']
    val_acc = history.history['val_r2_score']
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
    
    fig.savefig(f'./plots/training_{percent}_of_data.png')
    plt.close()

# # Shape and Statistics

depth_start = 4700
depth_end = 6420
depth_step = 20
depth_values = np.arange(depth_start, depth_end , depth_step)
num_ticks = 6  # Adjust the number of ticks as needed
depth_indices = np.linspace(0, len(depth_values) - 1, num_ticks, dtype=int)


phi_mature[phi_mature<0] = 0.0001
phi_max=np.max(phi_mature) #can also take 1 or critical porosity (0.4)
phi_min=np.min(phi_mature) #can also take 0

phi_scaled = scale_to_range(phi_mature)
seis_normalized = scale_to_range(seis_mature)

well_seismic_data = scale_to_range(well_seismic_data)
well_porosity_data = scale_to_range(well_porosity_data)


# # Experimento

results = pd.DataFrame(columns=['Porcentaje', 'Iteraciones', 'R2_test', 'Loss_test', 'Time', 'RMSE_explo', 'MAE_explo', 'R2_explo'])

porcentajes = [1,2,3,4,5,10,15,20,25,30,40,50,75]

for porcentaje_entrenamiento in porcentajes:
    print(f'Entrenamiento para {porcentaje_entrenamiento}% de los datos')
    for iter in range(1):
        print(f'Entrenamiento {iter+1}')
        porcentaje_validacion= 10

        train_wells = int((seis_normalized.shape[0] * seis_normalized.shape[1]) * (porcentaje_entrenamiento/100))
        val_wells = int((seis_normalized.shape[0] * seis_normalized.shape[1]) * (porcentaje_validacion/100))

        coords_train = np.zeros((train_wells, 2))
        coords_val = np.zeros((val_wells, 2))


        # Create a mask to keep track of which indices have been used
        mask = np.ones(seis_normalized.shape[:2], dtype=bool)

        def extract_traces(numer_of_wells, train=True):
            x_seismic = []
            y_porosity = []
            for _ in range(numer_of_wells):
                random_il = np.random.randint(0, seis_normalized.shape[0])
                random_xl = np.random.randint(0, seis_normalized.shape[1])
                
                # Ensure we pick an unused index
                while not mask[random_il, random_xl]:
                    random_il = np.random.randint(0, seis_normalized.shape[0])
                    random_xl = np.random.randint(0, seis_normalized.shape[1])
                    
                if train:
                    coords_train[_] = np.array([random_il, random_xl])
                else:
                    coords_val[_] = np.array([random_il, random_xl])
                
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
        X_val, Y_val = extract_traces(val_wells, train=False)
            
        # Remaining data for test
        X_test = seis_normalized[mask]
        Y_test = phi_scaled[mask]
        
        X_train_own_cnn = X_train.reshape(-1,86,1,1)
        Y_train_own_cnn = Y_train.reshape(-1,86,1,1)

        X_val_own_cnn = X_val.reshape(-1,86,1,1)
        Y_val_own_cnn = Y_val.reshape(-1,86,1,1)
        
        X_test_own_cnn = X_test.reshape(-1,86,1,1)
        Y_test_own_cnn = Y_test.reshape(-1,86,1,1)
        
        print('---------------------------------------------------')
        
        print(f'X_train_final shape: {X_train_own_cnn.shape}')
        print(f'Y_train shape: {Y_train_own_cnn.shape}')
        print(f'X_test_final shape: {X_test_own_cnn.shape}')
        print(f'X_val_final shape: {X_val_own_cnn.shape}')
        
        
        tf.keras.backend.clear_session()
        gc.collect()
        
        # Acá empieza mi modelo propio
        input_shape = (86, 1, 1)
        model_own = simplified_cnn(input_shape)
        print(model_own.summary())

        custom_loss = CustomSeismicLoss(
            model_own,
            well_seismic_data,
            well_porosity_data,
            lambda_param=1
        )

        initial_lr = 0.001
        model_own.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
                            loss=custom_loss,
                            metrics=[R2Nicolas()])

        # Define callbacks for this fold
        


        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_r2_score',
            mode='max',  # since we want to maximize R2
            factor=0.1,
            patience=1,
            min_lr=1e-6,
            verbose=1  # This will print when learning rate changes
        )
        loss_callback = LossComponentCallback(custom_loss)

        pos = np.random.permutation(Y_train_own_cnn.shape[0])
        start_time = time.time()

        # Train the model
        history_own = model_own.fit(
            X_train_own_cnn[pos],Y_train_own_cnn[pos],
            validation_data=(X_val_own_cnn, Y_val_own_cnn),
            epochs=50,
            batch_size=512,
            shuffle=True,
            callbacks=[lr_scheduler, loss_callback]
        )
        end_time = time.time()

        # Calculate the elapsed time
        elapsed_time = np.round(end_time - start_time)
        
        visualizacion_resultados(history_own, epocas, porcentaje_entrenamiento)
        
        print('Evaluación del modelo con datos de test')
        loss, accuracy = model_own.evaluate(X_test_own_cnn, Y_test_own_cnn)

        # PREDICCION DE LA RED
        seis_normalized_exploration = scale_to_range(seis_exploration)

        X_exploracion = seis_normalized_exploration.reshape(-1, 86)

        X_exploracion = np.expand_dims(X_exploracion, axis=-1)
        print(f'Predicción del modelo en el bloque de exploración que tiene shape {X_exploracion.shape}')
        phi_pred_exploracion = model_own.predict(X_exploracion)
        phi_pred_exploracion = unscale_from_range(phi_pred_exploracion, phi_min, phi_max)
        
        rmse = np.sqrt(mean_squared_error(phi_exploration.ravel() , phi_pred_exploracion.ravel()))
        mae = mean_absolute_error(phi_exploration.ravel() , phi_pred_exploracion.ravel())
        r2 = r2_score(phi_exploration.ravel() , phi_pred_exploracion.ravel())

        phi_pred_exploracion = phi_pred_exploracion.reshape(143, 370, 86)
        
        
        current_results = {'Porcentaje':[porcentaje_entrenamiento], 
                           'Iteraciones':[iter+1], 
                           'R2_test':[accuracy], 
                           'Loss_test':[loss], 
                           'Time':[elapsed_time], 
                           'RMSE_explo':rmse, 
                           'MAE_explo': mae, 
                           'R2_explo':r2}
        
        print(current_results)
        results = pd.concat([results, pd.DataFrame(current_results)])
        
        fig, ax = plt.subplots(1,3, figsize = (15, 4))

        fig.suptitle(f'Inline 84 Wells CCS1 VW1. Training with {porcentaje_entrenamiento}% of traces.')

        im1 = ax[0].imshow(seis_exploration[84-13,:,:].T, cmap='Greys')
        ax[0].set_title('Seismic')
        ax[0].set_aspect('auto')
        ax[0].set_yticks(depth_indices)
        ax[0].set_yticklabels(depth_values[depth_indices])
        fig.colorbar(im1, ax=ax[0], shrink=1)

        im2 = ax[1].imshow(phi_pred_exploracion[84-13,:,:].T, vmin=0, vmax=0.3, cmap='jet')
        ax[1].set_title('Estimated Porosity')
        ax[1].set_aspect('auto')
        ax[1].set_yticks(depth_indices)
        ax[1].set_yticklabels(depth_values[depth_indices])
        fig.colorbar(im2, ax=ax[1], shrink=1)

        im3 = ax[2].imshow(phi_exploration[84-13,:,:].T, vmin=0, vmax=0.3, cmap='jet')
        ax[2].set_title('Ground Truth Porosity')
        ax[2].set_aspect('auto')
        ax[2].set_yticks(depth_indices)
        ax[2].set_yticklabels(depth_values[depth_indices])
        fig.colorbar(im3, ax=ax[2], shrink=1)

        fig.tight_layout()
        fig.savefig(f"./plots/section_predicted_inline_{porcentaje_entrenamiento}_of_data.pdf", format="pdf", bbox_inches="tight")
        plt.close()
        
        fig, axito = plt.subplots(1,1)

        im_otra = axito.imshow(phi_pred_exploracion[84-13,:,:].T, vmin=0, vmax=0.3, cmap='jet')
        axito.set_title('Estimated Porosity')
        axito.set_aspect('auto')
        axito.set_yticks(depth_indices)
        axito.set_yticklabels(depth_values[depth_indices])
        fig.colorbar(im_otra, ax=axito, shrink=1)

        fig.savefig(f"./plots/estimated_porosity_{porcentaje_entrenamiento}_of_data.png")
        plt.close()
        
        # Clear model explicitly and delete related variables
        del model_own
        del X_train, Y_train, X_val, Y_val, X_test, Y_test
        del X_train_own_cnn, X_test_own_cnn, X_val_own_cnn
        del seis_normalized_exploration, X_exploracion, phi_pred_exploracion

        # Force garbage collection to free up memory
        gc.collect()

        # Ensure TensorFlow clears GPU memory
        tf.keras.backend.clear_session()

        # Optionally, reset all TensorFlow variables (useful if issues persist)
        tf.compat.v1.reset_default_graph()

        # Force GPU memory cleanup
        gc.collect()
        
        
        

results.to_csv('results_loop_training.csv', index=False)