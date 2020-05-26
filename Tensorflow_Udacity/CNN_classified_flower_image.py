# os package is used to read files and directory structure
import os
#numpy is used to convert python list to numpy array and to perform required matrix operations
import numpy as np
import glob
import shutil

import tensorflow as tf
#matplotlib.pyplot is used to plot the graph and display images in our training and validation data.
import matplotlib.pyplot as plt

##DATA LOADING
_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

zip_file = tf.keras.utils.get_file(origin=_URL,
                                   fname="flower_photos.tgz",
                                   extract=True)

base_dir = os.path.join(os.path.dirname(zip_file), 'flower_photos')

##The dataset we downloaded contains images of 5 types of flowers
classes = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']
#the dataset we have downloaded has following directory structure.
#flower_photos
#|__ diasy
#|__ dandelion
#|__ roses
#|__ sunflowers
#|__ tulips

#The code below creates a train and a val folder each containing 5 folders (one for each type of flower). It then moves the images from the original folders to these new folders such that 80% of the images go to the training set and 20% of the images go into the validation set. In the end our directory will have the following structure:


#Since we don't delete the original folders, they will still be in our flower_photos directory, but they will be empty. The code below also prints the total number of flower images we have for each type of flower
for cl in classes:
  img_path = os.path.join(base_dir, cl)
  images = glob.glob(img_path + '/*.jpg')
  print("{}: {} Images".format(cl, len(images)))
  train, val = images[:round(len(images)*0.8)], images[round(len(images)*0.8):]

  for t in train:
    if not os.path.exists(os.path.join(base_dir, 'train', cl)):
      os.makedirs(os.path.join(base_dir, 'train', cl))
    shutil.move(t, os.path.join(base_dir, 'train', cl))

  for v in val:
    if not os.path.exists(os.path.join(base_dir, 'val', cl)):
      os.makedirs(os.path.join(base_dir, 'val', cl))
    shutil.move(v, os.path.join(base_dir, 'val', cl))

#set up the path for the training and validation sets
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

#flower_photos
#|__ diasy
#|__ dandelion
#|__ roses
#|__ sunflowers
#|__ tulips
#|__ train
#    |______ daisy: [1.jpg, 2.jpg, 3.jpg ....]
#    |______ dandelion: [1.jpg, 2.jpg, 3.jpg ....]
#    |______ roses: [1.jpg, 2.jpg, 3.jpg ....]
#    |______ sunflowers: [1.jpg, 2.jpg, 3.jpg ....]
#    |______ tulips: [1.jpg, 2.jpg, 3.jpg ....]
# |__ val
#    |______ daisy: [507.jpg, 508.jpg, 509.jpg ....]
#    |______ dandelion: [719.jpg, 720.jpg, 721.jpg ....]
#    |______ roses: [514.jpg, 515.jpg, 516.jpg ....]
#    |______ sunflowers: [560.jpg, 561.jpg, 562.jpg .....]
#    |______ tulips: [640.jpg, 641.jpg, 642.jpg ....]


#Data Augmentation
#In tf.keras we can implement this using the same ImageDataGenerator class we used before. We can simply pass different transformations we would want to our dataset as a form of arguments and it will take care of applying it to the dataset during our training process.
#Experiment with Various Image Transformations
BATCH_SIZE = 100
IMG_SHAPE = 150

##Apply Random Horizontal Flip
#In the cell below, use ImageDataGenerator to create a transformation that rescales the images by 255 and then applies a random horizontal flip. Then use the .flow_from_directory method to apply the above transformation to the images in our training set. Make sure you indicate the batch size, the path to the directory of the training images, the target size for the images, and to shuffle the images.
image_gen = ImageDataGenerator(rescale=1./255,horizontal_flip= True)

train_data_gen= image_gen.flow_from_directory(batch_size=BATCH_SIZE,
                                              directory=train_dir,
                                              shuffle=True,
                                              target_size=(IMG_SHAPE,
                                              IMG_SHAPE))

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

##Apply Random rotaion
image_gen = ImageDataGenerator(rescale=1./255,rotation_range= 45)

train_data_gen = image_gen.flow_from_directory(batch_size=BATCH_SIZE,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=(IMG_SHAPE,IMG_SHAPE))

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

image_gen = ImageDataGenerator(rescale=1./255, zoom_range=0.5)

train_data_gen = image_gen.flow_from_directory(batch_size=BATCH_SIZE,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=(IMG_SHAPE, IMG_SHAPE))
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

##Puting all together
#use ImageDataGenerator to create a transformation that rescales the images by 255 and that applies:

#    random 45 degree rotation
#    random zoom of up to 50%
#    random horizontal flip
#    width shift of 0.15
#    height shift of 0.15

image_gen_train=ImageDataGenerator(rescale=1./255,
                                   rotation_range=45,
                                   width_shift_range=0.15,
                                   height_shift_range=0.15,
                                   shear_range=0.2,
                                   zoom_range=0.5,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
train_data_gen=image_gen_train.flow_from_directory(batch_size=BATCH_SIZE,                                                   directory=train_dir,
                                                   shuffle=True,
                                                   target_size=(IMG_SHAPE,IMG_SHAPE),
                                                   class_mode='sparse')
##Create a Data Generator for the Validation Set
image_gen_val=ImageDataGenerator(rescale=1./255)

val_data_gen = image_gen_val.flow_from_directory(batch_size=BATCH_SIZE,
                                                 directory=val_dir,
                                                 target_size=(IMG_SHAPE,IMG_SHAPE),class_mode='sparse')

##MODEL CREATION
model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),padding='same',activation='relu',
    input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32,(3,3),padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64,(3,3),padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(5,activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
              from_logits=True),metrics=['accuracy']
              )
#model using the fit_generator function instead of the usual fit function. We have to use the fit_generator function because we are using the ImageDataGenerator class to generate batches of training and validation data for our model. Train the model for 80 epochs and make sure you use the proper parameters in the fit_generator function.
epochs = 80

history = model.fit_generator( train_data_gen,
    steps_per_epoch=int(np.ceil(train_data_gen.n / float(batch_size))),
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(val_data_gen.n / float(batch_size)))
)

#Plot Training and Validation Graphs
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)
