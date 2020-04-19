import tensorflow as tf
import mitdeeplearning as mdl

import numpy as np
import matplotlib.pyplot as plt

#Automatic Differentiation in tensorflow
#Automatic is one of the most important part of tensorflow and is the backbone of
#training with backpropagation

#all forward-pass operations get recorded to a "tape"; then, to
#compute the gradient, the tape is played backwards

### Gradient computation with GradientTape ###

# y = x^2
# Example: x = 3.0
x = tf.Variable(3.0)

# Initiate the gradient tape
with tf.GradientTape() as tape:
  # Define the function
  y = x * x
# Access the gradient -- derivative of y with respect to x
dy_dx = tape.gradient(y, x)

assert dy_dx.numpy() == 6.0

#in training neural networks we use Differentiation and stochastic gradient descent
#to optimize a loss function
### Function minimization with automatic differentiation and SGD ###
##################################################################################
# Initialize a random value for our initial x
x = tf.Variable([tf.random.normal([1])])
print("Initializing x={}".format(x.numpy()))

learning_rate = 1e-2 # learning rate for SGD
history = []
# Define the target value
x_f = 4

# We will run SGD for a number of iterations. At each iteration, we compute the loss,
#   compute the derivative of the loss with respect to x, and perform the SGD update.
for i in range(500):
  with tf.GradientTape() as tape:
    #'''TODO: define the loss as described above'''
    loss = (x-x_f)**2

  # loss minimization using gradient tape
  grad = tape.gradient(loss, x) # compute the derivative of the loss with respect to x
  new_x = x - learning_rate*grad # sgd update
  x.assign(new_x) # update the value of x
  history.append(x.numpy()[0])

# Plot the evolution of x as we optimize towards x_f!
plt.plot(history)
plt.plot([0, 500],[x_f,x_f])
plt.legend(('Predicted', 'True'))
plt.xlabel('Iteration')
plt.ylabel('x value')

plt.show()
