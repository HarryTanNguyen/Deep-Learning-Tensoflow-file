#Lab 2: Computer Vision(Convolution Neural Network)
#In the first portion of this lab, we will build and train a convolutional #neural network (CNN) for classification of handwritten digits from the #famous MNIST dataset. The MNIST dataset consists of 60,000 training images #and 10,000 test images. Our classes are the digits 0-9.

import tensorflow as tf
import mitdeeplearning as mdl
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
assert len(tf.config.list_physical_devices('GPU')) > 0
#1.1 MNIST dataset
mnist = tf.keras.datasets.mnist
#Download data and display some samples from it
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = (np.expand_dims(train_images, axis=-1)/255.).astype(np.float32)
train_labels = (train_labels).astype(np.int64)

test_images = (np.expand_dims(test_images, axis=-1)/255.).astype(np.float32)


#Visualize some image and their corresponding traing labels look like
plt.figure(figsize=(10,10))
random_inds = np.random.choice(60000,36)
for i in range(36):
    plt.subplot(6,6,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    image_ind = random_inds[i]
    plt.imshow(np.squeeze(train_images[image_ind]), cmap=plt.cm.binary)
    plt.xlabel(train_labels[image_ind])
    #plt.show()

#To define the architecture of this first fully connected neural network
#we use Flatten layer to flatten the input so that in can be fed into the model
#build fully connected model
def build_fc_model():
    fc_model = tf.keras.Sequential([
        #First define a Flatten layer
        #transform the format of the image (28x28 pixels) to 1d array 28*28=784 pixels
        tf.keras.layers.Flatten(),

        #the first dense layer has 128 nodes
        #Define the activation function for the first fully connected (Dense) layer.
        tf.keras.layers.Dense(128,activation=tf.nn.relu),

        #Define the second Dense layer to output the classification probabilities
        tf.keras.layers.Dense(10,activation=tf.nn.softmax)
    ])
    return fc_model
model=build_fc_model()
#Before training the model, we need to define a few more settings. These are added during the model's compile step:

#Loss function — This defines how we measure how accurate the model is during training. As was covered in lecture, during training we want to minimize this function, which will "steer" the model in the right direction.
#Optimizer — This defines how the model is updated based on the data it sees and its loss function.
#Metrics — Here we can define metrics used to monitor the training and testing steps. In this example, we'll look at the accuracy, the fraction of the images that are correctly classified.

#Compile the model
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-1),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

##Train the model
# Define the batch size and the number of epochs to use during training
BATCH_SIZE = 64
EPOCHS = 5

# we can accomplish the training by calling the fit method
model.fit(train_images, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS)

## Evaluate accuracy on the test dataset
#nowe we can ask it to make prediction about the test set that it hasn't seen before
test_loss,test_acc=model.evaluate(test_images,test_labels)
print('Test_accuracy: ',test_acc)


###########################################################################
#1.3 Convolutional Neural Network for handwritten digit classification
#Define the CNN model
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
#initialize the model by passing some data through
cnn_model.predict(train_images[[0]])
#print the summary of the layers in the fc_modelprint(cnn_model.summary)
print(cnn_model.summary())

##Train and test the CNN model
cnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

cnn_model.fit(train_images,train_labels,batch_size=BATCH_SIZE,epochs=EPOCHS)

#evaluate metho to test the model
test_loss,test_acc=cnn_model.evaluate(test_images,test_labels)
print(test_acc)

##Make predicitions with the CNN model
predictions = cnn_model.predict(test_images)

print(predictions[0])

#a prediction is an array of 10 numbers
prediction= np.argmax(predictions[0])
print(prediction)



print("Label of this digit is:", test_labels[0])
plt.imshow(test_images[0,:,:,0], cmap=plt.cm.binary)

# visualize the classification results on the MNIST dataset. We will plot images from the test dataset along with their predicted label, as well as a histogram that provides the prediction probabilities for each of the digits:
image_index = 79 #@param {type:"slider", min:0, max:100, step:1}
plt.subplot(1,2,1)
mdl.lab2.plot_image_prediction(image_index, predictions, test_labels, test_images)
plt.subplot(1,2,2)
mdl.lab2.plot_value_prediction(image_index, predictions,  test_labels)
#We can also plot several images along with their predictions, where correct prediction labels are blue and incorrect prediction labels are red. The number gives the percent confidence (out of 100) for the predicted label. Note the model can be very confident in an incorrect prediction!
num_rows = 5
num_cols = 4
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  mdl.lab2.plot_image_prediction(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  mdl.lab2.plot_value_prediction(i, predictions, test_labels)
