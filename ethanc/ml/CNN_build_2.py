import tensorflow as tf
import keras
#from keras.models import Sequential
from keras import layers
#from tensorflow.keras.initializers import HeNormal
#from tensorflow.keras.regularizers import l2



weight_decay = 1e-4


def BUILD_MODEL_L2():
    model = keras.models.Sequential()
  
    model.add(layers.Conv1D(filters=20, kernel_size=3, padding='same', input_shape=(435,1), 
                            kernel_initializer=keras.initializers.HeNormal(), 
                            kernel_regularizer=keras.regularizers.l2(weight_decay)))
    model.add(layers.LeakyReLU(alpha=0.01))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(2))
    
    model.add(layers.Conv1D(filters=40, kernel_size=3, padding='same', 
                            kernel_initializer=keras.initializers.HeNormal(), 
                            kernel_regularizer=keras.regularizers.l2(weight_decay)))
    model.add(layers.LeakyReLU(alpha=0.01))
    model.add(layers.MaxPooling1D(2))
    
    model.add(layers.Dropout(0.1))  # Moderate dropout after increasing complexity
    
    model.add(layers.Conv1D(filters=80, kernel_size=3, padding='same', 
                            kernel_initializer=keras.initializers.HeNormal(), 
                            kernel_regularizer=keras.regularizers.l2(weight_decay)))
    model.add(layers.LeakyReLU(alpha=0.01))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(2))
    
    model.add(layers.Conv1D(filters=160, kernel_size=3, padding='same', 
                            kernel_initializer=keras.initializers.HeNormal(), 
                            kernel_regularizer=keras.regularizers.l2(weight_decay)))
    model.add(layers.LeakyReLU(alpha=0.01))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(2))
    
    model.add(layers.Dropout(0.2))  # Increased dropout in deeper layers
    
    model.add(layers.Conv1D(filters=320, kernel_size=3, padding='same', 
                            kernel_initializer=keras.initializers.HeNormal(), 
                            kernel_regularizer=keras.regularizers.l2(weight_decay)))
    model.add(layers.LeakyReLU(alpha=0.01))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(2))
    
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.2))  # Dropout before the final Dense layer
    
    model.add(layers.Dense(1, kernel_regularizer=keras.regularizers.l2(weight_decay)))
    
    model.summary()
    return model