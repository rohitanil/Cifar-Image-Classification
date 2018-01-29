"""Necessary library imports"""
import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.models import load_model

"""Function to load CIFAR dataset, normalize pixel values between 0.0 and 1.0, one hot encode target variables"""
def load_and_normalize():
    (Xtrain, Ytrain), (Xtest, Ytest) = cifar10.load_data()
    
    Xtrain=Xtrain.astype('float')
    Xtest=Xtest.astype('float')
    
    Xtrain=Xtrain/255.0
    Xtest=Xtest/255.0
    
    Ytrain = np_utils.to_categorical(Ytrain)
    Ytest = np_utils.to_categorical(Ytest)
    num_classes = Ytest.shape[1]
    return Xtrain,Ytrain,Xtest,Ytest,num_classes

"""Function to build a 5 layer CNN with 32,64,128,256 and 512 filters. Uses ReLU as activation function for all layers
except output layer(uses softmax function). Training the model for 50 epochs and 100 as batch size"""

def BuildModel():
    model= Sequential()
    model.add(Conv2D(32, (3, 3),input_shape=(3,32,32),padding="same", activation="relu"))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    
    model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(128, (3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(256, (3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    
    model.add(Conv2D(512, (3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    
    model.add(Flatten())
    model.add(Dense(1024,activation="relu"))
    model.add(Dropout(0.5))
   
    model.add(Dense(num_class,activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    model.fit(xtrain,ytrain,epochs=50, batch_size=100)
    model.save('image_classification.h5')

"""Function to load saved model and test it using test dataset """
def testModel(model_name):
    model=load_model(model_name)
    scores = model.evaluate(xtest, ytest, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    
""" main function """
if __name__ == "__main__":
    xtrain,ytrain,xtest,ytest,num_class=load_and_normalize()
    BuildModel()
    testModel('image_classification.h5')

