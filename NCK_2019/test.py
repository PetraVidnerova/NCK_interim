import sys
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping
# from keras.utils import plot_model
from sklearn.model_selection import StratifiedKFold, LeaveOneOut

import __main__ as main

from data import load_data
from model import createNetwork
import config as cfg 


def simple_test(X, C, y):

    # create network model                                              
    network = createNetwork()                                           
                                                                        
    network.summary()                                                   
    # plot_model doesn't work on haklnv (missing graphviz)              
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
                verbose=1,
                #callbacks=[EarlyStopping(monitor='acc', min_delta=0, patience=100, verbose=1)]
    )


    # calculate accuracy
    score = network.evaluate({'image_input': X, 'centroid_input': C}, y, verbose=0)
    print('Final loss:', score[0])
    print('Final accuracy:', 100*score[1], '%')

    #y_pred = network.predict({'image_input': X, 'centroid_input': C})
    #print("Accuracy: ", 100*sum(np.rint(y_pred.ravel()) == y.ravel())/len(y) ) 




def crossvalidation(X, C, y):
    # define 10-fold cross validation 
    # crossval = StratifiedKFold(n_splits=10, shuffle=True)
    crossval = LeaveOneOut()

    cvscores = []
    for train, test in crossval.split(X, y):
        # create model
        model = createNetwork()
        # Compile model
        model.compile(optimizer='rmsprop', 
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        # Fit the model
        model.fit({'image_input': X[train], 'centroid_input': C[train]},
                    y[train],
                    epochs=cfg.epochs, 
                    batch_size=cfg.batch_size,
                    verbose=0)

        # evaluate the model
        scores = model.evaluate({'image_input': X[test], 'centroid_input': C[test]}, y[test], verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))



if __name__ == "__main__":

    # ! not needed anymore: use CUDA_VISIBLE_DEVICES 
    # force Keras to use only one GPU
    # num_GPU = 1 
    # config = tf.ConfigProto(allow_soft_placement=True,
    #                        device_count = {'GPU' : num_GPU})
    # session = tf.Session(config=config)
    # K.set_session(session)
    
    
    # load and create data 
    # X, C ... inputs 
    # y .... outputs 
    X, C, y = load_data(format=K.image_data_format())
    
    #simple_test(X, C, y) 
    #crossvalidation(X, C, y)
    
    # choose a function to run according the first argument
    name = sys.argv[1]
    getattr(main, name)(X, C, y)
