"""
Homework Three
Maria Paula Baron Rodriguez 
Wsu ID: J858Q278
"""
import sys
print(sys.executable)


import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import os
import shutil
from sklearn.model_selection import train_test_split

#import cv2
#import scipy.stats as sp
#import random
#import re
#import numpy as np

plt.close('all')

Results_3 = r"C:\Users\Paula\.spyder-py3\MariaBaron-CS898BA-Project1\MariaBaron-CS898BA-Project1\Results_"

if os.path.exists(Results_3):shutil.rmtree(Results_3)
os.makedirs(Results_3)


# %% Part 2.1
Dataset_Path = (r"C:\Users\Paula\.spyder-py3\MariaBaron-CS898BA-Project1"r"\MariaBaron-CS898BA-Project1\Fish")

print(os.listdir(Dataset_Path))

Image_Paths = []
Image_Labels = []

for Class_Name in sorted(os.listdir(Dataset_Path)):

    Class_Path = os.path.join(Dataset_Path,Class_Name)
    
    for Filename in sorted(os.listdir(Class_Path)):

      Image_Path = os.path.join(Class_Path,Filename)

      Image_Paths.append(Image_Path)
      Image_Labels.append(Class_Name)

Dataset_DataFrame = pd.DataFrame({"Image_Path": Image_Paths,"Label": Image_Labels})

print("\nTotal number of images:")
print(len(Dataset_DataFrame))

print("\nNumber of classes:")
print(Dataset_DataFrame["Label"].nunique())

print("\nImages per class:")
print(    Dataset_DataFrame["Label"].value_counts().sort_index())

Train_DataFrame, Temporary_DataFrame = train_test_split(Dataset_DataFrame,test_size=0.30,random_state=42,stratify=Dataset_DataFrame["Label"])
Validation_DataFrame, Test_DataFrame = train_test_split(Temporary_DataFrame,test_size=0.50,random_state=42,stratify=Temporary_DataFrame["Label"])
Train_DataFrame = Train_DataFrame.reset_index(drop=True)

Validation_DataFrame = (Validation_DataFrame.reset_index(drop=True))
Test_DataFrame = Test_DataFrame.reset_index(drop=True)

Total_Images = len(Dataset_DataFrame)

Training_Percentage = (len(Train_DataFrame)/ Total_Images* 100)
Validation_Percentage = (len(Validation_DataFrame)/ Total_Images* 100)
Testing_Percentage = (len(Test_DataFrame)/ Total_Images* 100)


print("Training images:",len(Train_DataFrame),Train_DataFrame["Label"].value_counts().sort_index(),f"({Training_Percentage:.2f}%)")
print("Validation images:",len(Validation_DataFrame),Validation_DataFrame["Label"].value_counts().sort_index(),f"({Validation_Percentage:.2f}%)")
print("Testing images:",len(Test_DataFrame),Test_DataFrame["Label"].value_counts().sort_index(),f"({Testing_Percentage:.2f}%)")
print("Part 2.1 done")

# %%PART 2.2
Image_Height = 128
Image_Width = 128
Batch_Size = 32

Class_Names = sorted(Train_DataFrame["Label"].unique())

Class_To_Index = {Class_Name: Index 
                  for Index, Class_Name in enumerate(Class_Names)}

Train_DataFrame["Label_Index"] = (Train_DataFrame["Label"].map(Class_To_Index))
Validation_DataFrame["Label_Index"] = (Validation_DataFrame["Label"].map(Class_To_Index))
Test_DataFrame["Label_Index"] = (Test_DataFrame["Label"].map(Class_To_Index))

def Load_And_Preprocess_Image(Image_Path, Label):

    Image = tf.io.read_file(Image_Path)
    Image = tf.io.decode_image(Image,channels=3,expand_animations=False)
    Image = tf.image.resize(Image,[Image_Height, Image_Width])
    Image = tf.cast(Image,tf.float32)
    Image = Image / 255.0
    return Image, Label

def Create_Dataset(DataFrame,Shuffle=False):
    
    Image_Paths = DataFrame["Image_Path"].values
    Labels = DataFrame["Label_Index"].values
    
    Dataset = tf.data.Dataset.from_tensor_slices((Image_Paths,Labels))
    Dataset = Dataset.map(Load_And_Preprocess_Image,num_parallel_calls=tf.data.AUTOTUNE)

    if Shuffle:
        Dataset = Dataset.shuffle(buffer_size=len(DataFrame),seed=42)
    Dataset = Dataset.batch(Batch_Size)
    Dataset = Dataset.prefetch(tf.data.AUTOTUNE)
    
    return Dataset

Train_Dataset = Create_Dataset(Train_DataFrame,Shuffle=True)
Validation_Dataset = Create_Dataset(Validation_DataFrame,Shuffle=False)
Test_Dataset = Create_Dataset(Test_DataFrame,Shuffle=False)

for Images, Labels in Train_Dataset.take(1):
    print("Image batch shape:", Images.shape)
    print("Minimum pixel value:",tf.reduce_min(Images).numpy())
    print("Maximum pixel value:",tf.reduce_max(Images).numpy())
    
print("Part 2.2 done")
# %% PART 2.3
from tensorflow.keras import layers
Data_Augmentation = tf.keras.Sequential([layers.RandomFlip("horizontal"),layers.RandomRotation(0.08),layers.RandomBrightness(0.15)])

for Images, Labels in Train_Dataset.take(1):
    Augmented_Images = Data_Augmentation(Images,training=True)
    print("Augmented image batch shape:", Augmented_Images.shape)
print("Part 2.3 done")
