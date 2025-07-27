import os
import pandas as pd
import numpy as np
import PIL
import seaborn as sns
import pickle
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow import keras
from tf.keras.applications import DenseNet121
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from IPython.display import display
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from keras import optimizers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Change directory to your project folder
os.chdir('C:/Users/Vanshika.Bharadwaj/OneDrive - Ricoh Europe PLC/Documents/Final Year Project Ideas/EmotionAI/')
# Load facial key points data
keyfacialimage_df = pd.read_csv('C:/Users/Vanshika.Bharadwaj/OneDrive - Ricoh Europe PLC/Documents/Final Year Project Ideas/EmotionAI/data.csv')
print(keyfacialimage_df)
# Obtain relevant information about the DataFrame
keyfacialimage_df.info()
# Check if null values exist in the DataFrame
null_values = keyfacialimage_df.isnull().sum()
print(null_values)
# Check the shape of the 'Image' column
image_column_shape = keyfacialimage_df['Image'].shape
print(image_column_shape)
# Convert the 'Image' column values to 2D numpy arrays
keyfacialimage_df['Image'] = keyfacialimage_df['Image'].apply(lambda x: np.fromstring(x, dtype=int, sep=' ').reshape(96, 96))
# Obtain the shape of the first image
image_shape = keyfacialimage_df['Image'][0].shape
print(image_shape)
# Obtain a summary of the DataFrame
summary = keyfacialimage_df.describe()
print(summary)

import random
import numpy as np
import matplotlib.pyplot as plt
import copy

# Function to plot a random image with facial keypoints
def plot_random_image(df):
    i = np.random.randint(0, len(df))
    plt.imshow(df['Image'][i], cmap='gray')
    for j in range(1, 31, 2):
        plt.plot(df.loc[i][j-1], df.loc[i][j], 'rx')
    plt.pause(0.001)  # Pause to display the plot

# Function to plot a grid of images with facial keypoints
def plot_image_grid(df, num_images, grid_size):
    fig = plt.figure(figsize=(20, 20))
    for i in range(num_images):
        k = random.randint(0, len(df) - 1)
        ax = fig.add_subplot(grid_size, grid_size, i + 1)
        plt.imshow(df['Image'][k], cmap='gray')
        for j in range(1, 31, 2):
            plt.plot(df.loc[k][j-1], df.loc[k][j], 'rx')
    plt.pause(0.001)  # Pause to display the plot

# Plot a random image
plot_random_image(keyfacialimage_df)

# Plot a 4x4 grid of images
plot_image_grid(keyfacialimage_df, 16, 4)

# Plot an 8x8 grid of images
plot_image_grid(keyfacialimage_df, 64, 8)

# Task 4: Create a new copy of the DataFrame
keyfacialimage_df_copy = copy.deepcopy(keyfacialimage_df)

# Obtain the columns in the DataFrame, excluding the last column
columns = keyfacialimage_df_copy.columns[:-1]
print(columns)

# Horizontal Flip - flip the images along y axis
keyfacialimage_df_copy['Image'] = keyfacialimage_df_copy['Image'].apply(lambda x: np.flip(x, axis=1))

# Adjust x-coordinates after horizontal flip
for i in range(len(columns)):
    if i % 2 == 0:
        keyfacialimage_df_copy[columns[i]] = keyfacialimage_df_copy[columns[i]].apply(lambda x: 96. - float(x))

# Show the original image
plt.imshow(keyfacialimage_df['Image'][0], cmap='gray')
for j in range(1, 31, 2):
    plt.plot(keyfacialimage_df.loc[0][j-1], keyfacialimage_df.loc[0][j], 'rx')

plt.show()

# Show the horizontally flipped image
plt.imshow(keyfacialimage_df_copy['Image'][0], cmap='gray')
for j in range(1, 31, 2):
    plt.plot(keyfacialimage_df_copy.loc[0][j-1], keyfacialimage_df_copy.loc[0][j], 'rx')

plt.show()

# Continue running the code
# Example: Show another image or perform additional analysis
plt.imshow(keyfacialimage_df['Image'][1], cmap='gray')
for j in range(1, 31, 2):
    plt.plot(keyfacialimage_df.loc[1][j-1], keyfacialimage_df.loc[1][j], 'rx')

plt.show()

# Concatenate the original dataframe with the augmented dataframe along rows
augmented_df = pd.concat([keyfacialimage_df, keyfacialimage_df_copy], ignore_index=True)

# Check the shape of the augmented dataframe
print(augmented_df.shape)

# Randomly increasing the brightness of the images
# We multiply pixel values by random values between 1.5 and 2 to increase the brightness of the image
# We clip the value between 0 and 255

keyfacialimage_df_copy = copy.copy(keyfacialimage_df)
keyfacialimage_df_copy['Image'] = keyfacialimage_df_copy['Image'].apply(lambda x: np.clip(random.uniform(1.5, 2) * x, 0.0, 255.0))

# Concatenate the original dataframe with the augmented dataframe
augmented_df = pd.concat([augmented_df, keyfacialimage_df_copy], ignore_index=True)

# Check the shape of the augmented dataframe
print(augmented_df.shape)

# Show the image with increased brightness
plt.imshow(keyfacialimage_df_copy['Image'][0], cmap='gray')
for j in range(1, 31, 2):
    plt.plot(keyfacialimage_df_copy.loc[0][j-1], keyfacialimage_df_copy.loc[0][j], 'rx')

plt.show()

# Show the image with increased brightness
plt.imshow(keyfacialimage_df_copy['Image'][0], cmap='gray')
for j in range(1, 31, 2):
    plt.plot(keyfacialimage_df_copy.loc[0][j-1], keyfacialimage_df_copy.loc[0][j], 'rx')

plt.show()

# Obtain the value of images which is present in the 31st column (since index start from 0, we refer to 31st column by 30)
img = augmented_df.iloc[:, 30].values

# Normalize the images
img = img / 255.0

# Create an empty array of shape (x, 96, 96, 1) to feed the model
X = np.empty((len(img), 96, 96, 1))

# Iterate through the img list and add image values to the empty array after expanding its dimension from (96, 96) to (96, 96, 1)
for i in range(len(img)):
    X[i,] = np.expand_dims(img[i], axis=2)

# Convert the array type to float32
X = np.asarray(X).astype(np.float32)

# Check the shape of the array
print(X.shape)

# Obtain the value of x & y coordinates which are to be used as target
y = augmented_df.iloc[:, :30].values

# Convert the array type to float32
y = np.asarray(y).astype(np.float32)

# Check the shape of the array
print(y.shape)

from sklearn.model_selection import train_test_split

# Split the data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shapes of the resulting arrays
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Check the shape of the X_train array
print(X_train.shape)

# Check the shape of the X_test array
print(X_test.shape)

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Activation, Add
from tensorflow.keras.initializers import glorot_uniform

def res_block(X, filters, stage):
    # Convolutional block
    X_copy = X
    f1, f2, f3 = filters

    # Main Path
    X = Conv2D(f1, (1, 1), strides=(1, 1), name=f'res_{stage}_conv_a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = MaxPool2D((2, 2))(X)
    X = BatchNormalization(axis=3, name=f'bn_{stage}_conv_a')(X)
    X = Activation('relu')(X)

    X = Conv2D(f2, kernel_size=(3, 3), strides=(1, 1), padding='same', name=f'res_{stage}_conv_b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=f'bn_{stage}_conv_b')(X)
    X = Activation('relu')(X)

    X = Conv2D(f3, kernel_size=(1, 1), strides=(1, 1), name=f'res_{stage}_conv_c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=f'bn_{stage}_conv_c')(X)

    # Short path
    X_copy = Conv2D(f3, kernel_size=(1, 1), strides=(1, 1), name=f'res_{stage}_conv_copy', kernel_initializer=glorot_uniform(seed=0))(X_copy)
    X_copy = MaxPool2D((2, 2))(X_copy)
    X_copy = BatchNormalization(axis=3, name=f'bn_{stage}_conv_copy')(X_copy)

    # Add
    X = Add()([X, X_copy])
    X = Activation('relu')(X)

    # Identity Block 1
    X_copy = X

    # Main Path
    X = Conv2D(f1, (1, 1), strides=(1, 1), name=f'res_{stage}_identity_1_a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=f'bn_{stage}_identity_1_a')(X)
    X = Activation('relu')(X)

    X = Conv2D(f2, kernel_size=(3, 3), strides=(1, 1), padding='same', name=f'res_{stage}_identity_1_b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=f'bn_{stage}_identity_1_b')(X)
    X = Activation('relu')(X)

    X = Conv2D(f3, kernel_size=(1, 1), strides=(1, 1), name=f'res_{stage}_identity_1_c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=f'bn_{stage}_identity_1_c')(X)

    # Add
    X = Add()([X, X_copy])
    X = Activation('relu')(X)

    # Identity Block 2
    X_copy = X

    # Main Path
    X = Conv2D(f1, (1, 1), strides=(1, 1), name=f'res_{stage}_identity_2_a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=f'bn_{stage}_identity_2_a')(X)
    X = Activation('relu')(X)

    X = Conv2D(f2, kernel_size=(3, 3), strides=(1, 1), padding='same', name=f'res_{stage}_identity_2_b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=f'bn_{stage}_identity_2_b')(X)
    X = Activation('relu')(X)

    X = Conv2D(f3, kernel_size=(1, 1), strides=(1, 1), name=f'res_{stage}_identity_2_c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=f'bn_{stage}_identity_2_c')(X)

    # Add
    X = Add()([X, X_copy])
    X = Activation('relu')(X)

    return X

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout, ZeroPadding2D, BatchNormalization, Activation, Add
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform

def res_block(X, filters, stage):
    # Convolutional block
    X_copy = X
    f1, f2, f3 = filters

    # Main Path
    X = Conv2D(f1, (1, 1), strides=(1, 1), name=f'res_{stage}_conv_a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = MaxPool2D((2, 2))(X)
    X = BatchNormalization(axis=3, name=f'bn_{stage}_conv_a')(X)
    X = Activation('relu')(X)

    X = Conv2D(f2, kernel_size=(3, 3), strides=(1, 1), padding='same', name=f'res_{stage}_conv_b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=f'bn_{stage}_conv_b')(X)
    X = Activation('relu')(X)

    X = Conv2D(f3, kernel_size=(1, 1), strides=(1, 1), name=f'res_{stage}_conv_c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=f'bn_{stage}_conv_c')(X)

    # Short path
    X_copy = Conv2D(f3, kernel_size=(1, 1), strides=(1, 1), name=f'res_{stage}_conv_copy', kernel_initializer=glorot_uniform(seed=0))(X_copy)
    X_copy = MaxPool2D((2, 2))(X_copy)
    X_copy = BatchNormalization(axis=3, name=f'bn_{stage}_conv_copy')(X_copy)

    # Add
    X = Add()([X, X_copy])
    X = Activation('relu')(X)

    # Identity Block 1
    X_copy = X

    # Main Path
    X = Conv2D(f1, (1, 1), strides=(1, 1), name=f'res_{stage}_identity_1_a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=f'bn_{stage}_identity_1_a')(X)
    X = Activation('relu')(X)

    X = Conv2D(f2, kernel_size=(3, 3), strides=(1, 1), padding='same', name=f'res_{stage}_identity_1_b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=f'bn_{stage}_identity_1_b')(X)
    X = Activation('relu')(X)

    X = Conv2D(f3, kernel_size=(1, 1), strides=(1, 1), name=f'res_{stage}_identity_1_c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=f'bn_{stage}_identity_1_c')(X)

    # Add
    X = Add()([X, X_copy])
    X = Activation('relu')(X)

    # Identity Block 2
    X_copy = X

    # Main Path
    X = Conv2D(f1, (1, 1), strides=(1, 1), name=f'res_{stage}_identity_2_a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=f'bn_{stage}_identity_2_a')(X)
    X = Activation('relu')(X)

    X = Conv2D(f2, kernel_size=(3, 3), strides=(1, 1), padding='same', name=f'res_{stage}_identity_2_b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=f'bn_{stage}_identity_2_b')(X)
    X = Activation('relu')(X)

    X = Conv2D(f3, kernel_size=(1, 1), strides=(1, 1), name=f'res_{stage}_identity_2_c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=f'bn_{stage}_identity_2_c')(X)

    # Add
    X = Add()([X, X_copy])
    X = Activation('relu')(X)

    return X

input_shape = (96, 96, 1)

# Input tensor shape
X_input = Input(input_shape)

# Zero-padding
X = ZeroPadding2D((3, 3))(X_input)

# 1 - stage
X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
X = BatchNormalization(axis=3, name='bn_conv1')(X)
X = Activation('relu')(X)
X = MaxPooling2D((3, 3), strides=(2, 2))(X)

# 2 - stage
X = res_block(X, filters=[64, 64, 256], stage=2)

# 3 - stage
X = res_block(X, filters=[128, 128, 512], stage=3)

# Average Pooling
X = AveragePooling2D((2, 2), name='Average_Pooling')(X)

# Final layer
X = Flatten()(X)
X = Dense(4096, activation='relu')(X)
X = Dropout(0.2)(X)
X = Dense(2048, activation='relu')(X)
X = Dropout(0.1)(X)
X = Dense(30, activation='relu')(X)

model_1_facialKeyPoints = Model(inputs=X_input, outputs=X)
model_1_facialKeyPoints.summary()

import tensorflow as tf

# Define the Adam optimizer
adam = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)

# Compile the model
model_1_facialKeyPoints.compile(loss="mean_squared_error", optimizer=adam, metrics=['accuracy'])

from tensorflow.keras.callbacks import ModelCheckpoint

# Update the filepath to use the .keras extension
checkpointer = ModelCheckpoint(filepath="FacialKeyPoints_weights.keras", verbose=1, save_best_only=True)

history = model_1_facialKeyPoints.fit(X_train, y_train, 
                                      validation_data=(X_test, y_test), 
                                      epochs=1, 
                                      batch_size=32, 
                                      callbacks=[checkpointer])

history = model_1_facialKeyPoints.fit(X_train, y_train, 
                                      batch_size=32, 
                                      epochs=1, 
                                      validation_split=0.05, 
                                      callbacks=[checkpointer])

# Save the model architecture to a JSON file for future use
model_json = model_1_facialKeyPoints.to_json()
with open("FacialKeyPoints-model.json", "w") as json_file:
    json_file.write(model_json)


# Load the model architecture
with open('detection.json', 'r') as json_file:
    json_savedModel= json_file.read()

# load the model architecture
model_1_facialKeyPoints = tf.keras.models.model_from_json(json_savedModel)
model_1_facialKeyPoints.load_weights('weights_keypoint.hdf5')
adam = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model_1_facialKeyPoints.compile(loss="mean_squared_error", optimizer= adam , metrics = ['accuracy'])


# Evaluate the model on test data
result = model_1_facialKeyPoints.evaluate(X_test, y_test)

# Print the accuracy
print("Accuracy : {:.2f}%".format(result[1] * 100))

# Get the model keys
history.history.keys()

# Plot the training artifacts

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss','val_loss'], loc = 'upper right')
plt.show()


