import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.6):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True

#the fashion mnist data is available directly in the tf.keras dataset API
mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#if you want to see the value look like
import numpy as np
np.set_printoptions(linewidth=200)
import matplotlib.pyplot as plt
plt.imshow(training_images[0])
print(training_labels[0])
print(training_images[0])

callbacks = myCallback()

#create a neural network model
#Sequential:Define a sequence of layers in the neural network
#Flatten take a matrix and turn it into 1 dimension set
#Dense: adds a layer of neuron. Each layer of neurons need an activation funcion
#to tell them what to do
##Relu means only passes values 0 or greater to the next layer
##Softmax take a set of values and pick the biggest on for
##example if you have [0.1 0.1 0.05 9.5 0.05] after softmax
## u have [0 0 0 1 0]
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

#The next thing to do, now the model is defined, is to actually build it. You do
#this by compiling it with an optimizer and loss function as before --
#and then you train it by calling *model.fit * asking it to fit your training data
#to your training labels -- i.e. have it figure out the relationship
#between the training data and its actual labels, so in future if you have data that
#that looks like the training data, then it can make a prediction for
##what that data would look like. 
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])

model.evaluate(test_images, test_labels)
