import nibabel as nib
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
#from monai.transforms import LoadImage, Compose, ToTensor
#from monai.data import DataLoader, Dataset
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import tensorflow as tf 
from tensorflow import keras
from CNN_build import Build_model
#defining the tranform that will load the images
#can use this if I want to use any other transfroms that are available
#load_tf = LoadImage(image_only=True)

# Check if TensorFlow can detect the GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#dont want subjects in validation to end up in training
#disperse between grey and white and different subject
#stratify data 
#People that win dont have results 
#tell a story 
#
def load_tf(input):
    img = nib.load(input)
    return img.get_fdata()

def load_single_file(filepath, is_target=False):
    data = load_tf(filepath)
    if is_target:
        reshaped = data.reshape((-1, 1))
    else:
        reshaped = data.reshape((-1, data.shape[-1]))
    return reshaped

def load_files(inputdir, tardir):
    input_files = sorted([os.path.join(inputdir, file) for file in os.listdir(inputdir)])
    target_files = sorted([os.path.join(tardir, file) for file in os.listdir(tardir)])

    with ThreadPoolExecutor() as executor:
        infiles = list(executor.map(load_single_file, input_files))
        tarfiles = list(executor.map(lambda f: load_single_file(f, is_target=True), target_files))

    infiles = np.vstack(infiles)
    tarfiles = np.vstack(tarfiles)
    
    return infiles, tarfiles


input_dir = "/home/ethan.church/pre_ml/func/registered/main_data/training"   
target_dir = "/home/ethan.church/pre_ml/CVR_MAPS/registered/training"

print("beginning to load in files")
inputs, targets = load_files(input_dir, target_dir)
print("done loading in files")

#inputting into a dataset
#print(inputs.shape)
#print(targets.shape)
bold_train, bold_test, cvr_train, cvr_test = train_test_split(inputs,targets, test_size=0.1,train_size=0.9)

#need to remodel data to (batch_size,T,1)
bold_train = tf.expand_dims(tf.convert_to_tensor(bold_train),axis=2)
bold_train = tf.expand_dims(bold_train, axis=1)
cvr_train = tf.expand_dims(tf.convert_to_tensor(cvr_train), axis=2)
#print(bold_train.shape)
#print(bold_train.shape)
#rint(cvr_train.shape)

bold_test = tf.expand_dims(tf.convert_to_tensor(bold_test),axis=2)
bold_test = tf.expand_dims(bold_test, axis=1)
cvr_test = tf.expand_dims(tf.convert_to_tensor(cvr_test), axis=2)
test_data = tf.data.Dataset.from_tensor_slices((bold_test,cvr_test))

#inputting into dataset by taking rows from the tensors
data = tf.data.Dataset.from_tensor_slices((bold_train, cvr_train))
with tf.device('/GPU:0'):

    #defining model
    model = Build_model(X_train=bold_train)

    #compiling model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

    #training model
    history = model.fit(data,batch_size=50,epochs=5,shuffle=True,verbose=2,validation_data=test_data)

    model.save('GPU_run_model.keras')


#model = keras.models.load_model(filepath='first_run_model.',safemode=False)

#oss, accuracy = model.evaluate(test_data)
#print(f'loss on the test data was: {loss}')

