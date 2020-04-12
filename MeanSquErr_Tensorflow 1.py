import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
##NOTE: This template using tensorflow 1 keep in mind

#generation some house sizes between 1000 and 3500

num_house = 160
np.random.seed(42)
house_size=np.random.randint(low=1000,high=3500,size=num_house)

#generation house price from house size with a random noise added

np.random.seed(42)
house_price = house_size * 100 + np.random.randint(low=20000, high=70000, size = num_house)
##
##plt.plot(house_size,house_price,"bx")
##plt.ylabel("Price")
##plt.xlabel("Size")
##plt.show()

#normalize values to prevent under/overflow
def normalize(array):
    return(array - array.mean())/array.std()

#define number of training example,0.7 = 70%. We can take the first 70%
#since the values are random
num_train_sample = math.floor(num_house*0.7)

#define training data
train_house_size = np.asarray(house_size[:num_train_sample])
train_price = np.asarray(house_price[:num_train_sample])

train_house_size_norm=normalize(train_house_size)
train_price_norm=normalize(train_price)

#define test data
test_house_size= np.array(house_size[num_train_sample:])
test_house_price = np.array(house_price[num_train_sample:])

test_house_size_norm=normalize(test_house_size)
test_house_price_norm=normalize(test_house_price)


#tensor types:
#Constant - constant value
#Variable - values adjusted in graph
#PlaceHolder - used to pass data into graph

#set up the TensorFlow placeholders that get updated as we descend down the gradient
tf_house_size= tf.placeholder("float", name="house_size")
tf_price=tf.placeholder("float",name="price")

#define the variables holding the size_factor and price we set during training
#we initialize them to some random values based on the normal distribution
tf_size_factor=tf.Variable(np.random.randn(),name="size_factor")
tf_price_offset =tf.Variable(np.random.randn(),name="price_offset")

#2.Define the operation for the prediction valuse_ predicted price =(size_factor*house_size)+ price_offset
#Notice, the use of the tensorflow add and multiply function
tf_price_pred = tf.add(tf.multiply(tf_size_factor,tf_house_size),tf_price_offset)

#3. Define the Loss Function (how much error) - Mean squared error
tf_cost=tf.reduce_sum(tf.pow(tf_price_pred-tf_price,2))/(2*mum_train_samples)

#Optimizer learning rate. The size of the step down the gradient
learning_rate = 0.1

#4. Define a Gradient descent optimizer that will minimize the loss
#defined in the operation "cost"
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)


#initializing the varables
init= tf.global_variables_initializer()

#Launch the graph in the session
with tf.compat.v1.Session() as sess:
    sess.run(init)

    #set how of ten to display training progress and number of training iterations
    display_every=2
    num_training_iter=50

    #keep iterating the training data
    for iteration in range(num_training_iter):

        #Fit all training data
        for (x,y) in zip(train_house_size_norm,train_price_norm):
            sess.run(optimizer,feed_dict={tf_house_size: x,tf_price: y})

        #Display current status
        if(iteration+1)% display_every ==0:
            c=sess.run(tf_cost,feed_dict={tf_house_size:train_house_size_norm, tf_price:train_price_norm})
            print("iteration #:",'%04d' % (iteraion+1),"cost=","{:.9f}".format(c),\
                  "size_factor=",sess.run(tf_size_factor),"price_offset=",sees.run(tf_price_offset))
    print("Optimization Finished!")
    training_cost=sess.run(tf_cost, feed_dict={tf_house_size:train_house_size_norm, tf_price:train_price_norm})
    print("Trained cost=",training_cost,"size_factor=",sess.run(tf_size_factor),"price_offset=",sess.run(tf_price_offset),'\n')

    #plot training and test data and learned regression
    train_house_size_mean = train_house_size.mean()
    train_house_size_std=train_house_size.std()

    train_price_mean= train_price_mean()
    train_price_std= train_price.std()

    #plot graph
    plt.rcParams["figure.figsize"]=(10,8)
    plt.figure()
    plt.ylabel("price")
    plt.xlabel("size")
    plt.plot(train_house_size,train_price,'go',label='training data')
    plt.plot(test_house_size,test_house_price,'mo',label='Testing data')
    plt.plot(train_house_size_norm*train_house_size_std+train_house_size_mean,
             (sess.run(tf_size_factor)*train_house_size_norm+sess.run(tf_price_offset))*train_price_std+train_price_mean,
             label='Learned regression')
    
    plt.legent(loc='upper left')
    plt.show()
