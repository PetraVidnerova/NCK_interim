import numpy as np
import tensorflow as tf
from keras import backend as K
#from keras.utils import plot_model

from data import load_data
from model import createNetwork
import config as cfg 

# force Keras to use only one GPU
num_GPU = 1 
config = tf.ConfigProto(allow_soft_placement=True,
                        device_count = {'GPU' : num_GPU})
session = tf.Session(config=config)
K.set_session(session)


# load and create data 
# X, C ... inputs 
# y .... outputs 
X, C, y = load_data()

# create network model
network = createNetwork() 

network.summary()
# plot_model doesn't work on haklnv (graphviz missing)
#plot_model(network, to_file='network1.png')


# prepare for learning 
network.compile(optimizer='rmsprop', 
                loss='binary_crossentropy',
                metrics=['accuracy'])


# finaly train
network.fit({'image_input': X, 'centroid_input': C},
            y,
            epochs=cfg.epochs, 
            batch_size=cfg.batch_size,
            verbose=1)

# calculate accuracy
score = network.evaluate({'image_input': X, 'centroid_input': C}, y, verbose=0)
print('Final loss:', score[0])
print('Final accuracy:', 100*score[1], '%')

#y_pred = network.predict({'image_input': X, 'centroid_input': C})
#print("Accuracy: ", 100*sum(np.rint(y_pred.ravel()) == y.ravel())/len(y) ) 

