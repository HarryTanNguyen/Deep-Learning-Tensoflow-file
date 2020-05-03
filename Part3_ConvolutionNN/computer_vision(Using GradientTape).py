#CNN using tf.gradienttape
import tensorflow as tf
import mitdeeplearning as mdl
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
assert len(tf.config.list_physical_devices('GPU')) > 0

def build_cnn_model():
    cnn_model = tf.keras.Sequential([

    #define the first convolutional layer
    tf.keras.layers.Conv2D(filters=24,kernel_size=(3,3),activation=tf.nn.relu),

    #define the first max pooling layer
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),

    #define the second convolutional layer
    tf.keras.layers.Conv2D(filters=36,kernel_size=(3,3),activation=tf.nn.relu),

    #define the second max pooling layers
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation=tf.nn.relu),

    #output dense layer
    tf.keras.layers.Dense(10,activation=tf.nn.softmax)
    ])

    return cnn_model

cnn_model=build_cnn_model()

batch_size=12
loss_history=mdl.util.LossHistory(smoothing_factor=0.95) #to record the evolution of the loss_history
plotter=mdl.util.PeriodicPlotter(sec=2,xlabel='Iterations',ylabel='Loss',scale='semilogy')

optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2) # define our optimizer

if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists

for idx in tqdm(range(0, train_images.shape[0], batch_size)):
  # First grab a batch of training data and convert the input images to tensors
  (images, labels) = (train_images[idx:idx+batch_size], train_labels[idx:idx+batch_size])
  images = tf.convert_to_tensor(images, dtype=tf.float32)


# GradientTape to record differentiation operations
  with tf.GradientTape() as tape:
    #'''TODO: feed the images into the model and obtain the predictions'''
    logits = cnn_model(images)

    #'''TODO: compute the categorical cross entropy loss
    loss_value = tf.keras.backend.sparse_categorical_crossentropy() # TODO

  loss_history.append(loss_value.numpy().mean()) # append the loss to the loss_history record
  plotter.plot(loss_history.get())

  # Backpropagation
  '''TODO: Use the tape to compute the gradient against all parameters in the CNN model.
      Use cnn_model.trainable_variables to access these parameters.'''
  grads = tape.gradient(loss_value,cnn_model.trainable_variables)
  optimizer.apply_gradients(zip(grads, cnn_model.trainable_variables))
