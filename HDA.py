'''
This programe is based on MNIST database. It basically predict the handwritten digits.  
'''

#Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation 


#data load
data = pd.read_csv('train.csv')

data = np.array(data)
m, n = data.shape # 42000 X 785
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0] #lable
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255. 
_,m_train = X_train.shape


#Neural network math
def init_prams():
    w1 = np.random.rand(10,784) - 0.5 # Weight for the Input layer and 1st H-layer
    b1 = np.random.rand(10,1) - 0.5 # Bias for the Input layer and 1st H-layer

    w2 = np.random.rand(10,10) - 0.5 # Weight for the 1st H-layer and Output layer
    b2 = np.random.rand(10,1) - 0.5 # Bias for the 1st H-layer and Output layer
    
    return w1, w2, b1, b2

def ReLU(Z):
    return np.maximum(Z,0) #if Z is greter than 0 it returns Z else it returns 0

def softMax(Z):
    return np.exp(Z)/sum(np.exp(Z))

def forward_prop(w1, w2, b1, b2, X):
    Z1 = w1.dot(X) + b1
    A1 = ReLU(Z1)

    Z2 = w2.dot(A1) + b2 
    A2 = softMax(Z2)

    return Z1, Z2, A1, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max()+1))
    one_hot_Y[np.arange(Y.size), Y ] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_ReLU(Z):
    return Z > 0


def back_prop(Z1, Z2, A1, A2, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1/m * dZ2.dot(A1.T)
    db2 = 1/m * np.sum(dZ2)   

    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1/m * dZ1.dot(X.T)
    db1 = 1/m * np.sum(dZ1)

    return dW1, dW2, db1, db2

def update_prams(W1, W2, b1, b2, dW1, dW2, db1, db2, alpha):
    w1 = W1 - alpha*dW1
    w2 = W2 - alpha*dW2
    b1 = b1 - alpha*db1
    b2 = b2 - alpha*db2
    return w1, w2, b1, b2

def modal_predictions(A2):
    return np.argmax(A2,0)

def modal_accu(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions==Y)/Y.size

def gredient_decent(X, Y, iteration, alpha):
    w1, w2, b1, b2 = init_prams()
    for i in range(iteration):
        Z1, Z2, A1, A2 = forward_prop(w1, w2, b1, b2, X)
        dW1, dW2, db1, db2 = back_prop(Z1, Z2, A1, A2, w2, X, Y)
        w1, w2, b1, b2 = update_prams(w1, w2, b1, b2, dW1, dW2, db1, db2, alpha)
        if (i % 10 == 0):
            print("Iteration: ",i)
            print("Accuracy:", modal_accu(modal_predictions(A2),Y))
    return w1, w2, b1, b2

w1, w2, b1, b2 = gredient_decent(X_train, Y_train, 1000, 0.1)


def make_predictions(X, W1, b1, W2, b2):
    _,_,_,A2 = forward_prop(W1, W2, b1, b2, X)
    predictions = modal_predictions(A2)
    return predictions

def test_prediction(W1, b1, W2, b2, CI, label):
    prediction = make_predictions(CI, W1, b1, W2, b2)
    Label = label
    print("Prediction: ",prediction)
    print("Label: ",Label)

    current_image = CI.reshape((28, 28))*255
    title = f"Modal Prediction: {prediction}, Actual Value: [{Label}] "
    plt.title(title)
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.subplot().axis('off')
    plt.show()


while True:
    user_input = input('number: ')
    user_input = int(user_input)
    CI = X_train[:,user_input,None]
    label = Y_train[user_input]
    test_prediction(w1, b1, w2, b2, CI, label)   
