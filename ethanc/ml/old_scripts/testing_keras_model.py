import nibabel as nib
import numpy as np
from monai.transforms import LoadImage, Compose, ToTensor
#from monai.data import DataLoader, Dataset
import tensorflow as tf
from tensorflow import keras                                    
import os                                                      
from CNN_build import BUILD_MODEL_L2   
import matplotlib.pyplot as plt



# Function to load image using MONAI
def load_tf(input):
    loader = LoadImage(image_only=True)
    img = loader(input)
    return img

input_dir = "/home/ethan.church/pre_ml/func/registered/main_data/testing"   
target_dir = "/home/ethan.church/pre_ml/CVR_MAPS/registered/testing"

def load_files(inputdir, tardir):
    infiles = []
    tarfiles = []
    for file in os.listdir(inputdir):
        filepath = os.path.join(inputdir, file)
        a = load_tf(filepath)
        reshaped = a.reshape((-1, a.shape[-1]))
        infiles.append(reshaped)
    for file in os.listdir(tardir):
        filepath = os.path.join(tardir, file)
        b = load_tf(filepath)
        reshaped_cvr = b.reshape((-1, 1))
        tarfiles.append(reshaped_cvr)
    tarfiles = np.vstack(tarfiles)
    infiles = np.vstack(infiles)
    return infiles, tarfiles


print('starting loading in the data')
inputs, targets = load_files(input_dir, target_dir)
print('finished loading in data')


bold_train, cvr_train = inputs,targets

#need to remodel data to (batch_size,T,1)
print('reshaping the data')
bold_train = tf.expand_dims(tf.convert_to_tensor(bold_train),axis=2)
bold_train = tf.expand_dims(bold_train, axis=1)
cvr_train = tf.expand_dims(tf.convert_to_tensor(cvr_train), axis=2)
print(bold_train.shape[1:])
#print(bold_train.shape)
#rint(cvr_train.shape)

print('making a dataset')
training_data = tf.data.Dataset.from_tensor_slices((bold_train, cvr_train))


print('getting a validation sample')
input_dir = "/home/ethan.church/pre_ml/func/registered/main_data/Validation"   
target_dir = "/home/ethan.church/pre_ml/CVR_MAPS/registered/Validation"
bold_val, cvr_val = load_files(input_dir,target_dir)
#now need to pass into model as 
bold_val = tf.expand_dims(tf.convert_to_tensor(bold_val),axis=2)
bold_val = tf.expand_dims(bold_val, axis=1)
cvr_val = tf.expand_dims(tf.convert_to_tensor(cvr_val), axis=2)
validation_data = tf.data.Dataset.from_tensor_slices((bold_val, cvr_val))
print('finished splitting the validation')



#defining model
model = BUILD_MODEL_L2()

#compiling model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse','mae'])

#training model
history = model.fit(training_data,batch_size=32,epochs=20,shuffle=True,verbose=2,validation_data=validation_data)


# Extracting the training history
history_dict = history.history
# Plotting the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history_dict['loss'], label='Training Loss')
plt.plot(history_dict['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()


# Optional: Plotting the training and validation mean squared error (mse)
plt.figure(figsize=(10, 6))
plt.plot(history_dict['mse'], label='Training MSE')
plt.plot(history_dict['val_mse'], label='Validation MSE')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.title('Training and Validation Mean Squared Error')
plt.legend()
plt.grid(True)
plt.show()


model.save('Tuesday_morning_run.keras')
