import tensorflow as tf
import numpy as np 
import logging

logger =tf.get_logger()
logger.setLevel(logging.ERROR)

celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

for i,c in enumerate(celsius_q):
    print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))
    
model = tf.keras.Sequential([tf.keras.layers.Dense(units=1,input_shape=[1])])
#input_shape=[1] — This specifies that the input to this layer is a single value. That is, the shape is a one-dimensional array with one member. Since this is the first (and only) layer, that input shape is the input shape of the entire model. The single value is a floating point number, representing degrees Celsius.

#units=1 — This specifies the number of neurons in the layer. The number of neurons defines how many internal variables the layer has to try to learn how to solve the problem (more later). Since this is the final layer, it is also the size of the model's output — a single float value representing degrees Fahrenheit. (In a multi-layered network, the size and shape of the layer would need to match the input_shape of the next layer.)


model.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.Adam(0.1))
#Loss function — A way of measuring how far off predictions are from the desired outcome. (The measured difference is called the "loss

#Optimizer function — A way of adjusting internal values in order to reduce the loss.

history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training the model")
model.summary()

print(model.predict([100.0]))
#
#Feature: The input(s) to our model
#    Examples: An input/output pair used for training
#    Labels: The output of the model
#    Layer: A collection of nodes connected together within a neural network.
#    Model: The representation of your neural network
#    Dense and Fully Connected (FC): Each node in one layer is connected to each node in the previous layer.
#    Weights and biases: The internal variables of model
#    Loss: The discrepancy between the desired output and the actual output
#    MSE: Mean squared error, a type of loss function that counts a small number of large discrepancies as worse than a large number of small ones.
#    Gradient Descent: An algorithm that changes the internal variables a bit at a time to gradually reduce the loss function.
#    Optimizer: A specific implementation of the gradient descent algorithm. (There are many algorithms for this. In this course we will only use the “Adam” Optimizer, which stands for ADAptive with Momentum. It is considered the best-practice optimizer.)
#    Learning rate: The “step size” for loss improvement during gradient descent.
#    Batch: The set of examples used during training of the neural network
#    Epoch: A full pass over the entire training dataset
#    Forward pass: The computation of output values from input
#    Backward pass (backpropagation): The calculation of internal variable adjustments according to the optimizer algorithm, starting from the output layer and working back through each layer to the input.

#
