import pandas as pd
import numpy as np
import math
def load_mnist_data():
    dataset1=pd.read_csv("fashion-mnist_train.csv")
    y1=dataset1['label']
    y_train=np.array(y1).reshape(1,-1)
    xt1=[]
    for i in range(1,785):
        x1=dataset1['pixel'+str(i)]
        xt1.append(np.array(x1))
    x_t1=np.array(xt1) 
    dataset2=pd.read_csv("fashion-mnist_test.csv")
    y2=dataset2['label']
    y_test=np.array(y2).reshape(1,-1)
    xt2=[]
    for i in range(1,785):
       x1=dataset2['pixel'+str(i)]
       xt2.append(np.array(x1))
    x_t2=np.array(xt2)
    return x_t1.T.reshape(-1,28,28),y_train,x_t2.T.reshape(-1,28,28),y_test


def convert_to_one_hot(y,C):
    Y=np.eye(C)[y.reshape(-1)]
    return Y

def random_mini_batches(X,Y,minibatch_size):
    
    
    m=X.shape[0]
    mini_batches=[]
    
    permutation=list(np.random.permutation(m))
    
    shuffled_X=X[permutation,:,:]
    shuffled_Y=Y[permutation,:]
    
    
    number_of_minibatches=math.floor(m/minibatch_size)
    
    
    
    for k in range(number_of_minibatches):
        mini_batch_X=shuffled_X[k*minibatch_size:k*minibatch_size+minibatch_size,:,:]
        mini_batch_Y=shuffled_Y[k*minibatch_size:k*minibatch_size+minibatch_size,:]
        minibatch=(mini_batch_X,mini_batch_Y)
        mini_batches.append(minibatch)
        
    
    
    if m%minibatch_size != 0:
        mini_batch_X = shuffled_X[number_of_minibatches*minibatch_size:m,:,:]
        mini_batch_Y = shuffled_Y[number_of_minibatches*minibatch_size:m,:]
        minibatch=(mini_batch_X,mini_batch_Y)
    
    
    
    mini_batches.append(minibatch)
    return mini_batches