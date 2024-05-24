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

def Generator():
  inputs = tf.keras.layers.Input(shape=[32, 32, 1])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (batch_size, 16, 16, 64)
    downsample(128, 4),  # (batch_size, 8, 8, 128)
    downsample(256, 4),  # (batch_size, 4, 4, 256)
    downsample(512, 4),  # (batch_size, 2, 2, 512)
    downsample(512, 4),  # (batch_size, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(256, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(128, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(64, 4),  # (batch_size, 16, 16, 1024)
    upsample(64, 4),  # (batch_size, 32, 32, 512)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(1, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    print('-----------------------------------------------------------------------')
    print(down)
    x = down(x)
    print(x.shape)
    skips.append(x)
  print('--.--.--.-.--.--.--.--.-.--.--.--.--.-.--.--.--.--.-.--.--.--.--.-.--.')
  print(skips)
  skips = reversed(skips[:-1])
  print('--.--.--.-.--.--.--.--.-.--.--.--.--.-.--.--.--.--.-.--.--.--.--.-.--.')
  print(skips)
  
  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    print(f'El shape del upsample es{x.shape}')
    print(f'El shape del downsample es: {skip.shape}')
    x = tf.keras.layers.Concatenate()([x, skip])
  print('Sali sin problemas de la concatenación')
  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

LAMBDA = 100
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
  
    """La perdida del GAN está dada comparando con una matriz de 1s que sería una imagen perfectamente discrminada y la imagen que
    se generó"""
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    
    """Se saca el error absoluto medio MAE entre la imagen generada y la imagen objetivo"""
    # Mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    
    """La función de pérdida está dada por la perdida del discrminador mas la perdida del generador 
    ponderada con el factor Lambda de 100"""
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss

def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[32, 32, 1], name='input_image')
  tar = tf.keras.layers.Input(shape=[32, 32, 1], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x)  # (batch_size, 16, 16, 64)
  down2 = downsample(128, 4)(down1)  # (batch_size, 8, 8, 128)
  down3 = downsample(256, 4)(down2)  # (batch_size, 4, 4, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 6, 6, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 3, 3, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 5, 5, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 2, 2, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)

def discriminator_loss(disc_real_output, disc_generated_output):
  """Si el discrminador es 'perfecto' significa que su resultado será una matriz de 1s de las mismas dimensiones que la imagen.
  Es por esto que se crea una matriz de 1 y se compara con el resultado del discrminador a ver que tan lejos está con una función
  de binary cross entropy"""
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  """La matriz de 0s representa una imagen completamente falsa"""
  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  """La suma evalúa que tan bien lo está haciendo el discrminador"""
  total_disc_loss = real_loss + generated_loss

  return total_disc_loss

def generate_images(model, test_input, tar):
  print('Entré a generate images')
  prediction = model(test_input, training=True)
  print('salí de Generate Images')
  plt.figure(figsize=(15, 15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # Getting the pixel values in the [0, 1] range to plot.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()
  
def my_first_unet():

    X_train,X_val,X_test,Y_train,Y_val, Y_test = load_data()
    
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
    
    generator = Generator()
    discriminator = Discriminator()

    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)
    log_dir="logs/"

    summary_writer = tf.summary.create_file_writer(
      log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    @tf.function
    def train_step(input_image, target, step):
      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

      generator_gradients = gen_tape.gradient(gen_total_loss,
                                              generator.trainable_variables)
      discriminator_gradients = disc_tape.gradient(disc_loss,
                                                  discriminator.trainable_variables)

      generator_optimizer.apply_gradients(zip(generator_gradients,
                                              generator.trainable_variables))
      discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                  discriminator.trainable_variables))

      with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
        tf.summary.scalar('disc_loss', disc_loss, step=step//1000)
        
    def fit(train_ds, test_ds, steps):
      example_input, example_target = next(iter(test_ds.take(1)))
      print(example_input.shape)
      print(example_target.shape)
      start = time.time()

      for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
        if (step) % 1000 == 0:
          #display.clear_output(wait=True)

          if step != 0:
            print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

          start = time.time()

          generate_images(generator, example_input, example_target)
          print(f"Step: {step//1000}k")

        train_step(input_image, target, step)

        # Training step
        if (step+1) % 10 == 0:
          print('.', end='', flush=True)


        # Save (checkpoint) the model every 5k steps
        if (step + 1) % 5000 == 0:
          checkpoint.save(file_prefix=checkpoint_prefix)
    
      
    fit(train_dataset, test_dataset, steps=40000)
    
    # Restoring the latest checkpoint in checkpoint_dir
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    # Run the trained model on a few examples from the test set
    for inp, tar in test_dataset.take(5):
      generate_images(generator, inp, tar)
      
if __name__ =='__main__':
    my_first_unet()