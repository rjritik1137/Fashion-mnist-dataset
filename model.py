import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import functions
from tensorflow.python.framework import ops
train_X_orig,train_y_orig,test_X_orig,test_y_orig = load_mnist_data()
train_X=train_X_orig.reshape(-1,28,28,1)/255
test_X=test_X_orig.reshape(-1,28,28,1)/255
train_y=convert_to_one_hot(train_y_orig,10)
test_y=convert_to_one_hot(test_y_orig,10)




def model(train_X,train_y,test_X,test_y,learning_rate,epoch,batch_size):
    ops.reset_default_graph()
    (m,nx1,nx2,_)=train_X.shape
    n_y=train_y.shape[0]
    X,Y=create_placeholders(nx1,nx2,n_y)
    costs=[]
    channels=np.array([1,16,32,64])
    krnls=initialize_kernels(channels)
    logits=network(X,krnls)
    loss=tf.losses.softmax_cross_entropy(onehot_labels=Y,logits=logits)
    optimizer=tf.train.AdamOptimizer(0.0001).minimize(loss)
    
    init=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
       
        for epoch in range(epoch):
            epoch_cost=0
            minibatch_cost=0
            num_mini=int(m/batch_size)
            minibatch=random_mini_batches(train_X,train_y,100)
            for mini in minibatch:
                (mini_X,mini_y)=mini
                _,minibatch_cost=sess.run([optimizer,loss],feed_dict={X:mini_X,Y:mini_y})
                epoch_cost=epoch_cost+minibatch_cost/num_mini
            if epoch%100==0:
                print("Cost after epoch %i: %f" %(epoch,epoch_cost))
            if epoch%5==0:
                costs.append(epoch_cost)
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show() 
        correct_prediction = tf.equal(tf.argmax(logits,axis=1), tf.argmax(Y,axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print ("Train Accuracy:", accuracy.eval({X: train_X, Y: train_y}))
        print ("Test Accuracy:", accuracy.eval({X: test_X, Y: test_y}))               
    return logits
    



def network(X,krnls):
    inputs=tf.cast(tf.reshape(X,[-1,28,28,1]),tf.float32)    
    conv1=tf.nn.relu(tf.nn.conv2d(inputs,krnls["1"],padding="SAME",strides=[1,1,1,1])+krnls["b1"])
#    conv1=tf.layers.conv2d(inputs,filters=32,kernel_size=[3,3],strides=1,padding='same',activation='relu')
    pool1=tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2],strides=[2,2])
#    conv2=tf.layers.conv2d(inputs=pool1,filters=64,kernel_size=[3,3],strides=1,padding='same',activation='relu')
    conv2=tf.nn.relu(tf.nn.conv2d(pool1,krnls["2"],padding="SAME",strides=[1,1,1,1])+krnls["b2"])    
    pool2=tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],strides=[2,2])
#    conv3=tf.layers.conv2d(inputs=pool2,filters=128,kernel_size=[3,3],strides=1,padding='same',activation='relu')
    conv3=tf.nn.relu(tf.nn.conv2d(pool2,krnls["3"],padding="SAME",strides=[1,1,1,1])+krnls["b3"])
    pool3=tf.layers.max_pooling2d(inputs=conv3,pool_size=[2,2],strides=[2,2])
    
    flat=tf.reshape(pool3,[-1,pool3.shape[1]*pool3.shape[2]*pool3.shape[3]])
    dense1=tf.layers.dense(inputs=flat,units=512,activation='relu')
    dropout1=tf.layers.dropout(inputs=dense1,rate=0.5)
    dense2=tf.layers.dense(inputs=dropout1,units=256,activation='relu')
    dropout2=tf.layers.dropout(inputs=dense2,rate=0.5)
    logits=tf.layers.dense(inputs=dropout2,units=10)
    
    return logits


def create_placeholders(w,h,ny):
    x=tf.placeholder(tf.float32,[None,w,h,None],name='x')
    y=tf.placeholder(tf.int32,[None,10],name='y')
    return x,y



def initialize_kernels(channels):
    L=len(channels)

    parameter=dict()
    for i in range(1,L):
        parameter[str(i)]=tf.get_variable(name=str(i),shape=[3,3,channels[i-1],channels[i]],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        parameter["b"+str(i)]=tf.get_variable(name="b"+str(i),shape=[channels[i]],dtype=tf.float32,initializer=tf.zeros_initializer())
        print(parameter[str(i)].shape,parameter["b"+str(i)].shape)
        
    return parameter

