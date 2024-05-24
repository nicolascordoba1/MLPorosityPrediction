import os
import pathlib
import time
import datetime

import numpy as np

import tensorflow as tf

from sklearn import model_selection

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Switch to TkAgg backend (other options like QtAgg, WXAgg)

def plot_results(model, x, y):
    
    y_predict = model.predict(x)
    error = np.mean(np.square(y_predict-y)**2)
    
    print(f'El error es {error}')
    
    fig, ax = plt.subplots(1, 3)
    
    print(f'Las dimensiones del Y predict es {y_predict.shape}')
    print(f'Las dimensiones del Y normal es {y.shape}')
    
    ax[1].imshow(y_predict[1,:,:])
    ax[1].set_ylabel('Impedance')
    
    ax[1].imshow(y_predict[1,:,:])
    ax[1].set_ylabel('Porosity Modeled')
    
    ax[2].imshow(y[1,:,:])
    ax[2].set_ylabel('Porosity Original')
    
    plt.show()
    
def visualizacion_resultados(history, epocas):
    epochs = [i for i in range(epocas)]

    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']
    

    fig, ax = plt.subplots(1,2)

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


    plt.show()

def load_data():
  
    X_train = np.load('data_decatur/processed/train/impedance.npy')
    X_test = np.load('data_decatur/processed/test/impedance.npy')
    
    Y_train = np.load('data_decatur/processed/train/porosity.npy')
    Y_test = np.load('data_decatur/processed/test/porosity.npy')
    
    X_test,X_val,Y_test,Y_val = model_selection.train_test_split(X_test,Y_test, test_size=0.5, random_state=1,shuffle=False)
    
    number_train = np.random.random_integers(0,X_train.shape[0])
    
    number_test = np.random.random_integers(0,X_test.shape[0])
    
    fig, ax = plt.subplots(2,2)
    
    im = ax[0,0].imshow(X_train[number_train,:,:])
    ax[0,0].set_title('Impedance Train')
    fig.colorbar(im, ax=ax[0,0], shrink=0.8)
    
    im2 = ax[0,1].imshow(Y_train[number_train,:,:])
    ax[0,1].set_title('Porosity Train')
    fig.colorbar(im2, ax=ax[0,1], shrink=0.8)
    
    im3 = ax[1,0].imshow(X_test[number_test,:,:])
    ax[1,0].set_title('Impedance Test')
    fig.colorbar(im3, ax=ax[1,0], shrink=0.8)
    
    im4 = ax[1,1].imshow(Y_test[number_test,:,:])
    ax[1,1].set_title('Porosity Test')
    fig.colorbar(im4, ax=ax[1,1], shrink=0.8)

    plt.show()
    return X_train,Y_train,X_test,Y_test,X_val, Y_val

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    
    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

def unet_network():
  inputs = tf.keras.layers.Input(shape=[32, 32, 1])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (batch_size, 16, 16, 64)
    downsample(128, 4),  # (batch_size, 8, 8, 128)
    downsample(256, 4),  # (batch_size, 4, 4, 256)
    downsample(512, 4),  # (batch_size, 2, 2, 512)
    downsample(512, 4),  # (batch_size, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 512)
    upsample(256, 4, apply_dropout=True),  # (batch_size, 4, 4, 256)
    upsample(128, 4, apply_dropout=True),  # (batch_size, 8, 8, 128)
    upsample(64, 4),  # (batch_size, 16, 16, 64)
    upsample(64, 4),  # (batch_size, 32, 32, 64)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(1, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='sigmoid')  # (batch_size, 32, 32, 1)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)
    
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

def my_first_unet():

    X_train,Y_train,X_test,Y_test,X_val, Y_val = load_data()
    
    X_train = tf.expand_dims(X_train, axis=-1)
    X_val = tf.expand_dims(X_val, axis=-1)
    X_test = tf.expand_dims(X_test, axis=-1)
    
    Y_train = tf.expand_dims(Y_train, axis=-1)
    Y_val = tf.expand_dims(Y_val, axis=-1)
    Y_test = tf.expand_dims(Y_test, axis=-1)
    
    print(f'El shape del X_train es: {X_train.shape}')
    print(f'El shape del X_val es: {X_val.shape}')
    print(f'El shape del X_test es: {X_test.shape}')
    print(f'El shape del Y_train es: {Y_train.shape}')
    print(f'El shape del Y_val es: {Y_val.shape}')
    print(f'El shape del Y_test es: {Y_test.shape}')
    
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(X_train[1,:,:])
    ax[0].set_title('Impedance section')
    ax[1].imshow(Y_train[1,:,:])
    ax[1].set_title('Porosity section')
    fig.suptitle('Datos que van a entrar a entrenamiento ya separados')
    plt.show()
    
    train_images_dataset = tf.data.Dataset.from_tensor_slices(X_train)
    train_labels_dataset = tf.data.Dataset.from_tensor_slices(Y_train)
    test_images_dataset = tf.data.Dataset.from_tensor_slices(X_test)
    test_labels_dataset = tf.data.Dataset.from_tensor_slices(Y_test)
    validation_images_dataset = tf.data.Dataset.from_tensor_slices(X_val)
    validation_labels_dataset = tf.data.Dataset.from_tensor_slices(Y_val)
        
    train_dataset = tf.data.Dataset.zip((train_images_dataset, train_labels_dataset))
    test_dataset = tf.data.Dataset.zip((test_images_dataset, test_labels_dataset))
    validation_dataset = tf.data.Dataset.zip((validation_images_dataset, validation_labels_dataset))
    
    BUFFER_SIZE = 400
    BATCH_SIZE = 2000
    
    #train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    
    model = unet_network()
    print(model.summary())
    
    generator_optimizer = tf.keras.optimizers.Adam(1e-2)
    model.compile(optimizer=generator_optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])
    
    

    epocas = 1
    history_model = model.fit(x=X_train,
                              y=Y_train,
                              epochs=epocas,
                              batch_size=BATCH_SIZE,
                              validation_data=(X_val, Y_val),)
    
    model.save('unet_model.h5')
    
    visualizacion_resultados(history_model, epocas)
    
    plot_results(model, X_test, Y_test)

if __name__ =='__main__':
    my_first_unet()