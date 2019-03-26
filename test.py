#from keras.utils import plot_model

from data import load_data
from model import createNetwork

# X, C, y = load_data()

network = createNetwork() 

network.summary()
#plot_model(network, to_file='network1.png')
