from keras.models import Sequential, Model
from keras.layers import Input, Dense, concatenate, Conv2D, Flatten, MaxPooling2D, Dropout

import config as cfg 

def downSample(input_shape):
    
    # print("downSample input shape:", input_shape)

    model = Sequential([
        MaxPooling2D(pool_size=(4, 4), input_shape=input_shape),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(), 
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='sigmoid')
    ], name="DownSampler")

    #print("=================================================================")
    #print("=                    Down Sampler:                              =")
    #print("=================================================================")
    #model.summary()
    #print()

    return model 

def upSample(input_shape):
    
    model = Sequential([
        Dense(8, activation='relu', input_shape=input_shape),
        Dropout(0.5),
        Dense(16, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='sigmoid'),
    ], name="UpSampler")

    #print("=================================================================")
    #print("=                    Up Sampler:                                =")
    #print("=================================================================")
    #model.summary()
    #print()


    return model 

def mainNet(input_shape): 

    model = Sequential([
        Dense(32, activation='relu', input_shape=input_shape),
        Dropout(0.5),
        Dense(16, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid', name='main_output'),
    ], name="MainNetwork")

    return model 



def createNetwork(): 
    
    image_input = Input(shape=cfg.image_shape, name="image_input") 
    centroid_input = Input(shape=cfg.centroid_shape, name="centroid_input") 

    network1 = downSample(cfg.image_shape)(image_input)
    network2 = upSample(cfg.centroid_shape)(centroid_input) 
    main_input = concatenate([network1, network2])

    n_outputs = int(main_input.shape[1])
    main_output = mainNet((n_outputs,))(main_input)
    
    network = Model(input=[image_input, centroid_input],
                    output=main_output)
    
    return network 
