import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import ZeroPadding2D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.models import load_model

def load_and_normalize():
    (Xtrain, Ytrain), (Xtest, Ytest) = cifar10.load_data()
    
    Xtrain=Xtrain.astype('float')
    Xtest=Xtest.astype('float')
    
    Xtrain=Xtrain/255.0
    Xtest=Xtest/255.0
    
    YT=Ytest
    Ytrain = np_utils.to_categorical(Ytrain)
    Ytest = np_utils.to_categorical(Ytest)
    num_classes = Ytest.shape[1]
    return Xtrain,Ytrain,Xtest,Ytest,num_classes,YT

def BuildModel():
    model= Sequential()
    model.add(Conv2D(32, (3, 3),input_shape=(3,32,32),padding="same", activation="relu"))
    model.add(ZeroPadding2D(pool_size=(1,1)))
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
    model.fit(xtrain,ytrain,epochs=20, batch_size=100)
    model.save('image_classification.h5')

def testModel(model_name):
    model=load_model(model_name)
    pred= model.predict(xtest)
    pred=np.argmax(np.round(pred),axis=1)
    return pred

def printReports(predictions,original):
    print (classification_report(original,predictions))
    print ("Model Accuracy:",accuracy_score(original,predictions))


#def image_augmentation():
    #1. convert rgb to greyscale images
    #2. perform contour detection/ OTSU thresholding

if __name__ == "__main__":
    xtrain,ytrain,xtest,ytest,num_class,y=load_and_normalize()
    BuildModel()
    predictions=testModel('image_classification.h5')
    printReports(predictions,y)



