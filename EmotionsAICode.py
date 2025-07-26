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

# Load the model architecture from the JSON file
with open('detection.json', 'r') as json_file:
    json_savedModel = json_file.read()


# Load the model architecture
#model_1_facialKeyPoints = tf.keras.models.model_from_json(json_savedModel)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable(package='Custom', name='MyModel')
class MyModel(Model):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.dense = Dense(10)

    def call(self, inputs):
        return self.dense(inputs)

    def get_config(self):
        config = super(MyModel, self).get_config()
        return config

# Example usage
inputs = Input(shape=(96, 96, 1))
outputs = MyModel()(inputs)
model = Model(inputs, outputs)

# Load the model weights
model_1_facialKeyPoints.load_weights('weights_keypoint.hdf5')

# Define the Adam optimizer
adam = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)

# Compile the model
model_1_facialKeyPoints.compile(loss="mean_squared_error", optimizer=adam, metrics=['accuracy'])

# Evaluate the model
result = model_1_facialKeyPoints.evaluate(X_test, y_test)
print("Accuracy : {}".format(result[1]))

# Get the model keys
keys = history.history.keys()
print(keys)

""" # Plot the training artifacts
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'val_loss'], loc='upper right')
plt.show() """
# Check if history contains the required keys
if 'loss' in history.history and 'val_loss' in history.history:
    # Plot the training artifacts
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'val_loss'], loc='upper right')
    plt.show()
else:
    print("The history object does not contain 'loss' and 'val_loss' keys.")


# Read the CSV file for the facial expression data
facialexpression_df = pd.read_csv('icml_face_data.csv')

# Display the first few rows of the dataframe
print(facialexpression_df.head())

# Display the first few rows of the DataFrame
print(facialexpression_df.head())

# Display a summary of the DataFrame
print(facialexpression_df.info())

# Access the first element in the 'pixels' column
first_pixel_data = facialexpression_df[' pixels'][0]

# Display the data
print(first_pixel_data)

# Function to convert pixel values in string format to array format
def string2array(x):
    return np.array(x.split(' ')).reshape(48, 48, 1).astype('float32')

import cv2

# Function to resize images from (48, 48) to (96, 96)
def resize(x):
    img = x.reshape(48, 48)
    return cv2.resize(img, dsize=(96, 96), interpolation=cv2.INTER_CUBIC)

# Apply the string2array function to the 'pixels' column
facialexpression_df[' pixels'] = facialexpression_df[' pixels'].apply(lambda x: string2array(x))

# Apply the resize function to the 'pixels' column
facialexpression_df[' pixels'] = facialexpression_df[' pixels'].apply(lambda x: resize(x))

# Display the first few rows of the DataFrame
print(facialexpression_df.head())

# Check the shape of the DataFrame
print(facialexpression_df.shape)

# Check for the presence of null values in the DataFrame
null_values = facialexpression_df.isnull().sum()
print(null_values)

# Dictionary to map numerical labels to facial expressions
label_to_text = {0: 'anger', 1: 'disgust', 2: 'sad', 3: 'happiness', 4: 'surprise'}

import matplotlib.pyplot as plt

# Display the first image in the 'pixels' column
plt.imshow(facialexpression_df[' pixels'][0], cmap='gray')
plt.title('First Image')
plt.axis('off')  # Hide the axis
plt.show()

# List of emotions
emotions = [0, 1, 2, 3, 4]

# Loop through each emotion and display one image
for i in emotions:
    data = facialexpression_df[facialexpression_df['emotion'] == i][:1]
    img = data[' pixels'].item()
    img = img.reshape(96, 96)
    plt.figure()
    plt.title(label_to_text[i])
    plt.imshow(img, cmap='gray')
    plt.axis('off')  # Hide the axis for a cleaner look
    plt.show()

# Get the indices of unique emotion labels sorted by their frequency
emotion_indices = facialexpression_df.emotion.value_counts().index
print(emotion_indices)

# Get the count of each unique emotion label
emotion_counts = facialexpression_df.emotion.value_counts()
print(emotion_counts)

import matplotlib.pyplot as plt
import seaborn as sns

# Create a bar plot of the emotion counts
plt.figure(figsize=(10, 10))
sns.barplot(x=facialexpression_df.emotion.value_counts().index, y=facialexpression_df.emotion.value_counts())
plt.title('Emotion Counts')
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.show()

#TASK #14: PERFORM DATA PREPARATION AND IMAGE AUGMENTATION
# Split the dataframe into features and labels

# Split the dataframe into features and labels
from keras.utils import to_categorical

X = facialexpression_df[' pixels']
y = to_categorical(facialexpression_df['emotion'])

first_element = X[0]
print(first_element)
print(y)

# Stack the elements of X along a new axis
X = np.stack(X, axis=0)

# Reshape X to the desired dimensions
X = X.reshape(24568, 96, 96, 1)

# Print the shapes of X and y
print(X.shape, y.shape)

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)

# Split the testing set into validation and testing sets
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=True)

# Print the shapes of the datasets
print(X_train.shape, X_val.shape, X_test.shape)
print(y_train.shape, y_val.shape, y_test.shape)

print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)
print(X_train.shape, y_train.shape)

# Image pre-processing
X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0

# Print the first few elements of X_train
print(X_train[:5])

# Print the shape of X_train
print(X_train.shape)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create an ImageDataGenerator with extended data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[1.1, 1.5],
    fill_mode="nearest"
)


from tensorflow.keras.layers import Input, ZeroPadding2D, Conv2D, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D, Flatten, Dense, Add
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform

def res_block(X, filters, stage):
    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    # First component of main path
    X = Conv2D(F1, (1, 1), strides=(1, 1), name=f'res{stage}_branch2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=f'bn{stage}_branch2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(F2, (3, 3), strides=(1, 1), padding='same', name=f'res{stage}_branch2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=f'bn{stage}_branch2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(F3, (1, 1), strides=(1, 1), name=f'res{stage}_branch2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=f'bn{stage}_branch2c')(X)

    # Shortcut path
    X_shortcut = Conv2D(F3, (1, 1), strides=(1, 1), name=f'res{stage}_branch1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=f'bn{stage}_branch1')(X_shortcut)

    # Add shortcut value to main path
    X = Add()([X, X_shortcut])
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

# 4 - stage
# X = res_block(X, filters=[256, 256, 1024], stage=4)

# Average Pooling
X = AveragePooling2D((4, 4), name='Average_Pooling')(X)

# Final layer
X = Flatten()(X)
X = Dense(5, activation='softmax', name='Dense_final', kernel_initializer=glorot_uniform(seed=0))(X)

model_2_emotion = Model(inputs=X_input, outputs=X, name='Resnet18')

model_2_emotion.summary() 


# Compile the model
# model_2_emotion.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Assuming you have your training data in variables X_train and Y_train
# and your validation data in variables X_val and Y_val


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Early stopping to exit training if validation loss is not decreasing after certain epochs (patience)
earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

# Save the best model with the lowest validation loss
checkpointer = ModelCheckpoint(filepath="FacialExpression_weights.keras", verbose=1, save_best_only=True)

# Train the model with early stopping and model checkpointing
history = model_2_emotion.fit(X_train, y_train, epochs=2, batch_size=32, validation_data=(X_val, y_val),
                              callbacks=[earlystopping, checkpointer])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Assuming you have already defined and compiled your model as model_2_emotion

# Early stopping to exit training if validation loss is not decreasing after certain epochs (patience)
earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

# Save the best model with the lowest validation loss
checkpointer = ModelCheckpoint(filepath="FacialExpression_weights.keras", verbose=1, save_best_only=True)

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Train the model with early stopping and model checkpointing
history = model_2_emotion.fit(
    train_datagen.flow(X_train, y_train, batch_size=64),
    validation_data=(X_val, y_val),
    steps_per_epoch=len(X_train) // 64,
    epochs=2,
    callbacks=[checkpointer, earlystopping]
)

# Saving the model architecture to a JSON file
model_json = model_2_emotion.to_json()
with open("FacialExpression-model.json", "w") as json_file:
    json_file.write(model_json)

#Task 17 - ASSESS THE PERFORMANCE OF TRAINED FACIAL EXPRESSION CLASSIFIER MODEL

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Define any custom layers or classes if used in the model
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.python.keras.models import model_from_json

# Save model architecture to JSON file
model_json = model_2_emotion.to_json()
with open("emotion.json", "w") as json_file:
    json_file.write(model_json)

# Save model weights to HDF5 file
# model_2_emotion.save_weights("weights_emotions.hdf5")
model_2_emotion.save_weights("weights_emotions.weights.h5")


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.python.keras.models import model_from_json

@register_keras_serializable(package='Custom', name='CustomLayer')
class CustomLayer(Layer):
    def __init__(self, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)
    
    def call(self, inputs):
        return inputs

try:     
    # Load the model architecture from a JSON file
    with open('emotion.json', 'r') as json_file:
        json_savedModel = json_file.read()
    print("Model architecture loaded successfully.")

    # Define custom objects
    custom_objects = {
        'CustomLayer': CustomLayer  # Add other custom objects if needed
    }

    # Recreate the model from the JSON data
    model_2_emotion = model_from_json(json_savedModel, custom_objects=custom_objects)
    print("Model recreated from JSON successfully.")

    # Load the model weights
    model_2_emotion.load_weights('weights_emotions.hdf5')
    print("Model weights loaded successfully.")

    # Compile the model with optimizer, loss function, and metrics
    model_2_emotion.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])
    print("Model compiled successfully.")

    # Load test data (Ensure X_test and y_test are defined)
    # Example: Replace with actual test data loading method
    # X_test, y_test = load_test_data() 
    if 'X_test' not in locals() or 'y_test' not in locals():
        raise ValueError("Test data (X_test, y_test) is not defined. Load your test dataset before evaluation.")

    # Evaluate the model on the test data
    score = model_2_emotion.evaluate(X_test, y_test, verbose=0)
    print('Test Accuracy: {:.2f}%'.format(score[1] * 100))

    # Check if history object exists
    if 'history' in locals():
        # Extract training and validation accuracy and loss from the history object
        accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        # Plot training and validation accuracy
        epochs = range(len(accuracy))
        plt.plot(epochs, accuracy, 'bo', label='Training Accuracy')
        plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.show()

        # Plot training and validation loss
        plt.plot(epochs, loss, 'ro', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and Validation loss')
        plt.legend()
        plt.show()
    else:
        print("History object not found. Ensure the model has been trained and the history object is available.")

    # Predict classes for the test data
    predicted_classes = np.argmax(model_2_emotion.predict(X_test), axis=-1)
    y_true = np.argmax(y_test, axis=-1)
    print("Predictions made successfully.")

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, predicted_classes)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
    print("Confusion matrix plotted successfully.")

    # Define label_to_text mapping (replace with actual class labels)
    label_to_text = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4: "Surprised"}  

    # Display some test images with their predicted and true labels
    L, W = 5, 5  # Grid size for visualization
    fig, axes = plt.subplots(L, W, figsize=(12, 12))
    axes = axes.ravel()
    
    for i in np.arange(0, L * W):
        if i < len(X_test):  # Ensure we don't exceed test set size
            axes[i].imshow(X_test[i].reshape(96, 96), cmap='gray')
            axes[i].set_title(f'Pred: {label_to_text[predicted_classes[i]]}\nTrue: {label_to_text[y_true[i]]}')
            axes[i].axis('off')

    plt.subplots_adjust(wspace=1)
    plt.show()
    print("Test images displayed successfully.")

    # Print the classification report
    print(classification_report(y_true, predicted_classes))
    print("Classification report printed successfully.")

except FileNotFoundError as fnf_error:
    print("File not found error:", fnf_error)
except ValueError as val_error:
    print("Value error:", val_error)
except Exception as e:
    print("An error occurred:", e)


## TASK #18: COMBINE BOTH MODELS
# FACIAL KEY POINTS DETECTION AND FACIAL EXPRESSION MODELS

def predict(X_test):

  # Making prediction from the keypoint model
  df_predict = model_1_facialKeyPoints.predict(X_test)

  # Making prediction from the emotion model
  df_emotion = np.argmax(model_2_emotion.predict(X_test), axis=-1)

  # Reshaping array from (856,) to (856,1)
  df_emotion = np.expand_dims(df_emotion, axis = 1)

  # Converting the predictions into a dataframe
  df_predict = pd.DataFrame(df_predict, columns= columns)

  # Adding emotion into the predicted dataframe
  df_predict['emotion'] = df_emotion

  return df_predict

df_predict = predict(X_test)

df_predict.head()

# Plotting the test images and their predicted keypoints and emotions

fig, axes = plt.subplots(4, 4, figsize = (24, 24))
axes = axes.ravel()

for i in range(16):

    axes[i].imshow(X_test[i].squeeze(),cmap='gray')
    axes[i].set_title('Prediction = {}'.format(label_to_text[df_predict['emotion'][i]]))
    axes[i].axis('off')
    for j in range(1,31,2):
            axes[i].plot(df_predict.loc[i][j-1], df_predict.loc[i][j], 'rx')

import json
import tensorflow.keras.backend as K

def deploy(directory, model):
  MODEL_DIR = directory
  version = 1

  # Let's join the temp model directory with our chosen version number
  # The expected result will be = '\tmp\version number'
  export_path = os.path.join(MODEL_DIR, str(version))
  print('export_path = {}\n'.format(export_path))

  # Let's save the model using saved_model.save

  tf.saved_model.save(model, export_path)

  os.environ["MODEL_DIR"] = MODEL_DIR

  ## TASK #20. SERVE THE MODEL USING TENSORFLOW SERVING

  # TASK #21: MAKE REQUESTS TO MODEL IN TENSORFLOW SERVING


