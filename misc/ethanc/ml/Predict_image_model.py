import nibabel as nib
import numpy as np
from monai.transforms import LoadImage, Compose, ToTensor
#from monai.data import DataLoader, Dataset
import tensorflow as tf
import keras                                    
import os                                                      
from CNN_build_2 import BUILD_MODEL_L2   
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
'''
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
history = model.fit(training_data,batch_size=32,epochs=1,shuffle=True,verbose=2,validation_data=validation_data)
model.save('models/predicted_image_run_2.keras')

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
'''
model = keras.models.load_model(filepath="models/predicted_image_run_2.keras")
model.compile(optimizer='adam', loss='mean_squared_error')
print('making a predicted image on the validation data') #probably want only one image? pull one from the training dataset?
# can use the same methods to make an input
indir = '/home/ethan.church/pre_ml/func/registered/main_data/for_preds'
tardir ='/home/ethan.church/pre_ml/CVR_MAPS/registered/for_preds'
actual_in, actual_out = load_files(indir,tardir)
actual_in = tf.expand_dims(tf.convert_to_tensor(actual_in),axis=2)
actual_in = tf.expand_dims(actual_in, axis=1)
preds_data = tf.data.Dataset.from_tensor_slices(actual_in)
print(f"want shape to be (64x64x26,435) actual shape was {actual_in.shape}")
#now need to reshape the input directory, dont really need to for tardir since already in desired format
predictions = model.predict(preds_data,batch_size=32,verbose=2)
print(f"the shape of the prediction matrix is {predictions.shape}")

#now need to do the inverse of the reshaping 
predicted_image= np.reshape(predictions,(64,64,26))
# now just need a 2D slice
# Slice from the predicted image
slice2d = predicted_image[:,:,10]
actual_out= np.reshape(actual_out,(64,64,26))
slice2d_orig = actual_out[:,:,10]
# Create a figure with two subplots, side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image
axes[0].imshow(slice2d.T, cmap='gray', origin='lower')
axes[0].axis('off')

# Slice from the second image
axes[1].imshow(slice2d_orig.T, cmap='gray', origin='lower')
axes[1].axis('off')

# Save the figure with both images
output = 'output/predicted_image.png'
plt.savefig(output, bbox_inches='tight', pad_inches=0)

# Display the images
#plt.show()

