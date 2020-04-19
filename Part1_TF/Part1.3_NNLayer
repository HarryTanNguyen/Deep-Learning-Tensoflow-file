import mitdeeplearning as mdl
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


### Defining a neural network using the Sequential API ###

# Import relevant packages
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
# Define the number of outputs
n_output_nodes = 3

# First define the model
model = Sequential()

#'''TODO: Define a dense (fully connected) layer to compute z'''
# Remember: dense layers are defined by the parameters W and b!
# You can read more about the initialization of W and b in the TF documentation :)
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense?version=stable
dense_layer = Dense(n_output_nodes,activation='sigmoid')

# Add the dense layer to the model
model.add(dense_layer)
# Test model with example input
x_input = tf.constant([[1,2.]], shape=(1,2))

#'''TODO: feed input into the model and predict the output!'''
model_output = model(x_input)
print(model_output)

###############################################################################
#Define a model using subclassing
#Subclassing affords us a lot of flexibility to define custom models
#For example, we can use boolean arguments in the call function to specify different
#network behaviors, for example different behaviors during training and inference
class SubclassModel(tf.keras.Model):

  # In __init__, we define the Model's layers
  def __init__(self, n_output_nodes):
    super(SubclassModel, self).__init__()
    ##'''TODO: Our model consists of a single Dense layer. Define this layer.'''
    self.dense_layer = Dense(n_output_nodes,activation='sigmoid')

  # In the call function, we define the Model's forward pass.
  def call(self, inputs):
    return self.dense_layer(inputs)
n_output_nodes = 3
model = SubclassModel(n_output_nodes)

x_input = tf.constant([[1,2.]], shape=(1,2))

print(model.call(x_input))
################################################################################
#Subclassing affords us a lot of flexibility to define custom models
#For example, we can use boolean arguments in the call function to specify different
#network behaviors, for example different behaviors during training and inference
class IdentityModel(tf.keras.Model):

  # As before, in __init__ we define the Model's layers
  # Since our desired behavior involves the forward pass, this part is unchanged
  def __init__(self, n_output_nodes):
    super(IdentityModel, self).__init__()
    self.dense_layer = tf.keras.layers.Dense(n_output_nodes, activation='sigmoid')

  #'''TODO: Implement the behavior where the network outputs the input, unchanged, under control of the isidentity argument.'''
  def call(self, inputs, isidentity=False):
    x = self.dense_layer(inputs)
    if isidentity: # TODO
      return inputs # TODO
    return x
#Test IdentityModel


n_output_nodes = 3
model = IdentityModel(n_output_nodes)

x_input = tf.constant([[1,2.]], shape=(1,2))
#'''TODO: pass the input into the model and call with and without the input identity option.'''
out_activate = model.call(x_input) # TODO
# out_activate = # TODO
out_identity = model.call(x_input, isidentity=True) # TODO
# out_identity = # TODO

print("Network output with activation: {}; network identity output: {}".format(out_activate.numpy(), out_identity.numpy())
