import numpy as np

import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras import regularizers

from sklearn import model_selection
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Switch to TkAgg backend (other options like QtAgg, WXAgg)

def plot_results(model, x, y):
    
    y_predict = model.predict(x)
    error = np.mean(np.square(y_predict-y)**2)
    
    print(f'El error es {error}')
    
    fig, ax = plt.subplots(1, 2)
    
    print(f'Las dimensiones del Y predict es {y_predict.shape}')
    print(f'Las dimensiones del Y normal es {y.shape}')
    
    ax[0].imshow(y_predict[1,:,:])
    ax[0].set_ylabel('Porosity Modeled')
    
    ax[1].imshow(y[1,:,:])
    ax[1].set_ylabel('Porosity Original')
    
    plt.show()
    
def visualizacion_resultados(history, epocas):
    epochs = [i for i in range(epocas)]

    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    train_lr = history.history['lr']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']
    

    fig, ax = plt.subplots(1,2)
    #fig.set_size_inches(16,7)

    ax[0].plot(epochs, train_acc, 'go-', label='accuracy-train')
    ax[0].plot(epochs, val_acc, 'ro-', label='accuracy-val')
    ax[0].set_title('Accuracy train')
    ax[0].legend()
    ax[0].set_xlabel('epochs')
    ax[0].set_ylabel('accuracy')


    ax[1].plot(epochs, train_loss, 'go-', label='loss-train')
    ax[1].plot(epochs, val_loss, 'ro-', label='loss-val')
    ax[1].set_title('Loss train')
    ax[1].legend()
    ax[1].set_xlabel('epochs')
    ax[1].set_ylabel('loss')
    """ax[2].plot(epochs, train_lr, 'go-', label='lr-train')
    ax[2].set_title('Learning Rate train')
    ax[2].legend()
    ax[2].set_xlabel('epochs')
    ax[2].set_ylabel('Learning Rate')"""

    plt.show()

def load_data():
    impedance = np.load('data_poseidon/processed/impedance_blocked.npy')
    porosity = np.load('data_poseidon/processed/porosity_blocked.npy')
    
    X_train,X_test,Y_train,Y_test = model_selection.train_test_split(impedance,porosity, test_size=0.20, random_state=1,shuffle=False)
    
    X_train,X_val,Y_train,Y_val = model_selection.train_test_split(X_train,Y_train, test_size=0.25, random_state=1,shuffle=False)
    
    fig, ax = plt.subplots(1,2)
    
    im = ax[0].imshow(impedance[1,:,:])
    ax[0].set_title('Impedance section')
    fig.colorbar(im, ax=ax[0], shrink=0.8)
    
    im2 = ax[1].imshow(porosity[1,:,:])
    ax[1].set_title('Porosity section')
    fig.colorbar(im2, ax=ax[1], shrink=0.8)

    plt.show()
    
    return X_train,X_val,X_test,Y_train,Y_val, Y_test

def my_first_ann():

    X_train,X_val,X_test,Y_train,Y_val, Y_test = load_data()
    print(f'El tamaño de los datos de entrada de entrenamiento son: {X_train.shape}')
    print(f'El tamaño de los datos de entrada de test son: {X_test.shape}')
    print(f'El tamaño de los datos de entrada de validacion son: {X_val.shape}')
    
    print(f'El tamaño de los datos de salida de entrenamiento son: {Y_train.shape}')
    print(f'El tamaño de los datos de salida de test son: {Y_test.shape}')
    print(f'El tamaño de los datos de salida de validacion son: {Y_val.shape}')
    
    fig, ax = plt.subplots(1,2, figsize=(20,8))
    ax[0].imshow(X_train[1,:,:])
    ax[0].set_title('Impedance section')
    ax[1].imshow(Y_train[1,:,:])
    ax[1].set_title('Porosity section')
    fig.suptitle('Datos que van a entrar a entrenamiento ya separados')
    plt.show()
    
    inputs = tf.keras.Input(shape=(32,32))
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(1024, activation=tf.nn.tanh)(x)
    x = tf.keras.layers.Dense(1024, activation=tf.nn.tanh)(x)
    x = tf.keras.layers.Dense(1024, activation=tf.nn.tanh)(x)
    x = tf.keras.layers.Dense(1024, activation=tf.nn.tanh)(x)
    x = tf.keras.layers.Dense(1024, activation=tf.nn.tanh)(x)
    x = tf.keras.layers.Dense(1024, activation=tf.nn.tanh)(x)
    x = tf.keras.layers.Dense(1024, activation=tf.nn.tanh)(x)
    x = tf.keras.layers.Dense(1024, activation=tf.nn.tanh)(x)
    x = tf.keras.layers.Dense(1024, activation=tf.nn.tanh)(x)
    x = tf.keras.layers.Dense(1024, activation=tf.nn.tanh)(x)
    x = tf.keras.layers.Reshape((32,32))(x)
    outputs = tf.keras.layers.Dense(32, activation=tf.nn.sigmoid)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    print(model.summary())
    model.compile(optimizer=tf.keras.optimizers.experimental.Adagrad(),
              loss='mse',
              metrics=['accuracy'])
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint('model_naive.h5', 
                                                monitor='val_accuracy', 
                                                mode='max', 
                                                verbose=1, 
                                                save_weights_only=True)

    """callback_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='accuracy', 
                                                    factor=0.5, 
                                                    patience=5, 
                                                    min_lr=0.000001)"""
    epocas = 100
    
    model_ann = model.fit(X_train, 
            Y_train,
            epochs=epocas,
            callbacks=[checkpoint],
            validation_data=(X_val, Y_val),
            batch_size=5,
            )
    visualizacion_resultados(model_ann, epocas)
    plot_results(model, X_test, Y_test)

def my_first_cnn():
    X_train,X_val,X_test,Y_train,Y_val, Y_test = load_data()
    
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    Y_train = np.expand_dims(Y_train, axis=-1)
    Y_test = np.expand_dims(Y_test, axis=-1)
    X_val = np.expand_dims(X_val, axis=-1)
    Y_val = np.expand_dims(Y_val, axis=-1)
    
    base_filtros = 64
    #w_regularizer = 1e-4
    
    model_cnn = tf.keras.Sequential()

    #Conv 1
    model_cnn.add(L.Conv2D(base_filtros, (4,4), input_shape=X_train.shape[1:]))
    model_cnn.add(L.Activation('tanh'))
    
    #Conv 2
    model_cnn.add(L.Conv2D(base_filtros, (4,4)))
    model_cnn.add(L.Activation('tanh'))
    #Conv 2
    model_cnn.add(L.Conv2D(base_filtros, (4,4)))
    model_cnn.add(L.Activation('tanh'))
    #Conv 2
    model_cnn.add(L.Conv2D(base_filtros, (4,4)))
    model_cnn.add(L.Activation('tanh'))
    #Conv 2
    model_cnn.add(L.Conv2D(base_filtros, (4,4)))
    model_cnn.add(L.Activation('tanh'))
    #Conv 2
    model_cnn.add(L.Conv2D(base_filtros, (4,4)))
    model_cnn.add(L.Activation('tanh'))
    #Conv 2
    model_cnn.add(L.Conv2D(base_filtros, (4,4)))
    model_cnn.add(L.Activation('tanh'))

    #UpScaling

    model_cnn.add(L.Conv2DTranspose(32, (16,16), activation='tanh'))
    model_cnn.add(L.Conv2DTranspose(1, (4,4), activation='tanh'))
    model_cnn.add(L.Conv2DTranspose(1, (4,4), activation='tanh'))
    
    model_cnn.add(L.Dense(1))
    model_cnn.add(L.Activation('sigmoid'))

    print(model_cnn.summary())

    model_cnn.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint('model_naive.h5', 
                                                monitor='val_accuracy', 
                                                mode='max', 
                                                verbose=1, 
                                                save_weights_only=True)

    """callback_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='accuracy', 
                                                    factor=0.5, 
                                                    patience=10, 
                                                    min_lr=0.000001)"""
    epocas = 100
    
    cnn_history = model_cnn.fit(X_train, 
            Y_train,
            epochs=epocas,
            callbacks=[checkpoint],
            validation_data=(X_val, Y_val)
            )
    
    visualizacion_resultados(cnn_history, epocas)

if __name__ =='__main__':
    my_first_cnn()