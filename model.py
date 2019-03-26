from keras.models import Sequential, Model
from keras.layers import Input, Dense, concatenate, Conv2D, Flatten

import config as cfg 

def downSample(input_shape):
    
    # print("downSample input shape:", input_shape)

    model = Sequential([
        Conv2D(64, (3, 3), input_shape=input_shape, padding='same'),
        Flatten(), 
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
    ])

    return model 

def upSample(input_shape):
    
    model = Sequential([
        Dense(8, activation='relu', input_shape=input_shape),
        Dense(16, activation='relu'),
        Dense(32, activation='relu'),
    ])

    return model 

def mainNet(input_shape): 

    model = Sequential([
        Dense(32, activation='relu', input_shape=input_shape),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid', name='main_output'),
    ])

    return model 



def createNetwork(): 
    
    image_input = Input(shape=cfg.image_shape, name="image_input") 
    centroid_input = Input(shape=cfg.centroid_shape, name="centroid_input") 

    network1 = downSample(cfg.image_shape)(image_input)
    network2 = upSample(cfg.centroid_shape)(centroid_input) 
    main_input = concatenate([network1, network2])

    n_outputs = int(main_input.shape[1])
    main_network = mainNet((n_outputs,))

    main_output = main_network(main_input)

    
    network = Model(input=[image_input, centroid_input],
                    output=main_output)
    
    return network 
