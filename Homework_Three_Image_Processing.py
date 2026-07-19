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
import random
import numpy as np

plt.close('all')

Random_Seed = 42
random.seed(Random_Seed)
np.random.seed(Random_Seed)
tf.random.set_seed(Random_Seed)

Results_3 = r"C:\Users\Paula\.spyder-py3\MariaBaron-CS898BA-Project1\MariaBaron-CS898BA-Project1\Results_3"

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
Data_Augmentation = tf.keras.Sequential([layers.RandomFlip("horizontal"),layers.RandomRotation(10/360),layers.RandomBrightness(factor=0.15,value_range=(0.0, 1.0))])

for Images, Labels in Train_Dataset.take(1):
    Augmented_Images = Data_Augmentation(Images,training=True)
    print("Augmented image batch shape:", Augmented_Images.shape)
print("Part 2.3 done")

# %% PART 3.1
Number_Of_Classes = len(Class_Names)
Model = tf.keras.Sequential([tf.keras.layers.Input(shape=(Image_Height, Image_Width, 3)),
    Data_Augmentation,
    
# Part 3.1.1
    tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation="relu",padding="same"),
# PART 3.1.2
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    
# Part 3.1.1
    tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation="relu",padding="same"),
# PART 3.1.2
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    
# Part 3.1.1
    tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),activation="relu",padding="same"),
# PART 3.1.2
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

# PART 3.1.3
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128,activation="relu"),
    tf.keras.layers.Dense(units=Number_Of_Classes,activation="softmax")])

Model.summary()
print("Part 3.1 done")

# %% PART 3.2

Learning_Rate = 0.001
Epochs = 10

Optimizer = tf.keras.optimizers.Adam(learning_rate=Learning_Rate)
Model.compile(optimizer=Optimizer,loss="sparse_categorical_crossentropy",metrics=["accuracy"])
History = Model.fit(Train_Dataset,validation_data=Validation_Dataset,epochs=Epochs)

Model.save(os.path.join(Results_3,"Baseline_CNN_Model.keras"))
print("Model saved successfully.")

Epoch_Numbers = range(1,len(History.history["accuracy"]) + 1)
Figure, Axes = plt.subplots(1,2,figsize=(12, 5))
Axes[0].plot(Epoch_Numbers,History.history["loss"],label="Training")
Axes[0].plot(Epoch_Numbers,History.history["val_loss"],label="Validation")
Axes[0].set_title("Baseline CNN Loss")
Axes[0].set_xlabel("Epoch")
Axes[0].set_ylabel("Cross-Entropy Loss")
Axes[0].legend()
Axes[0].grid(alpha=0.3)

Axes[1].plot(Epoch_Numbers,History.history["accuracy"],label="Training")
Axes[1].plot(Epoch_Numbers,History.history["val_accuracy"],label="Validation")
Axes[1].set_title("Baseline CNN Accuracy")
Axes[1].set_xlabel("Epoch")
Axes[1].set_ylabel("Accuracy")
Axes[1].set_ylim(0, 1)
Axes[1].legend()
Axes[1].grid(alpha=0.3)


Figure.tight_layout()
Figure.savefig(os.path.join(Results_3,"Baseline_Training_Curves.png"),dpi=150)

plt.show()

# %% PART 4.1

Learning_Rates = [0.01, 0.001, 0.0001]
Batch_Sizes = [32, 64]
Dropout_Rates = [0.3, 0.5]

def Build_Model(Dropout_Rate):

    Model = tf.keras.Sequential([tf.keras.layers.Input(shape=(Image_Height, Image_Width, 3)),
        Data_Augmentation,

        tf.keras.layers.Conv2D(32,(3, 3),activation="relu",padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(64,(3, 3),activation="relu",padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(128,(3, 3), activation="relu",padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation="relu"),
        tf.keras.layers.Dropout(Dropout_Rate),
        tf.keras.layers.Dense(Number_Of_Classes,activation="softmax")])

    return Model

print("Part 4.1 done")

# %% PART 4.2

Tuning_Results = []

Best_Validation_Loss = float("inf")
Best_Configuration = None
Best_Model_Path = os.path.join(Results_3,"Best_Optimized_CNN_Model.keras")
Temporary_Model_Path = os.path.join(Results_3,"Temporary_Best_Model.keras")

for Learning_Rate_Value in Learning_Rates:

    for Batch_Size_Value in Batch_Sizes:

        for Dropout_Rate_Value in Dropout_Rates:

            print("\nLearning Rate:",Learning_Rate_Value,"| Batch Size:",Batch_Size_Value,"| Dropout Rate:",Dropout_Rate_Value)

            tf.keras.backend.clear_session()
            Batch_Size = Batch_Size_Value
            Train_Dataset_Tuning = Create_Dataset(Train_DataFrame,Shuffle=True)
            Validation_Dataset_Tuning = Create_Dataset(Validation_DataFrame,Shuffle=False)
            Tuning_Model = Build_Model(Dropout_Rate_Value)

            Tuning_Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=Learning_Rate_Value),
                loss="sparse_categorical_crossentropy",metrics=["accuracy"])

            Model_Checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=Temporary_Model_Path,
                monitor="val_loss",
                mode="min",
                save_best_only=True)

            Tuning_History = Tuning_Model.fit(
                Train_Dataset_Tuning,
                validation_data=Validation_Dataset_Tuning,
                epochs=Epochs,
                callbacks=[Model_Checkpoint],
                verbose=1)

            Minimum_Validation_Loss = min(Tuning_History.history["val_loss"])
            Best_Epoch = Tuning_History.history["val_loss"].index(Minimum_Validation_Loss) + 1
            Best_Validation_Accuracy = Tuning_History.history["val_accuracy"][Best_Epoch-1]

            Tuning_Results.append({
                "Learning_Rate":Learning_Rate_Value,
                "Batch_Size":Batch_Size_Value,
                "Dropout_Rate":Dropout_Rate_Value,
                "Validation_Loss":Minimum_Validation_Loss,
                "Validation_Accuracy":Best_Validation_Accuracy,
                "Best_Epoch":Best_Epoch
            })

            if Minimum_Validation_Loss < Best_Validation_Loss:

                Best_Validation_Loss = Minimum_Validation_Loss

                Best_Configuration = {
                    "Learning_Rate":Learning_Rate_Value,
                    "Batch_Size":Batch_Size_Value,
                    "Dropout_Rate":Dropout_Rate_Value,
                    "Validation_Loss":Minimum_Validation_Loss,
                    "Validation_Accuracy":Best_Validation_Accuracy,
                    "Best_Epoch":Best_Epoch
                }

                Best_Current_Model = tf.keras.models.load_model(Temporary_Model_Path)
                Best_Current_Model.save(Best_Model_Path)

                Best_Optimized_History = pd.DataFrame(Tuning_History.history)
                Best_Optimized_History.to_csv(os.path.join(Results_3,"Best_Optimized_Model_History.csv"),index=False)

if os.path.exists(Temporary_Model_Path):
    os.remove(Temporary_Model_Path)

print("Part 4.2 done")


# %% PART 4.3

Tuning_Results_DataFrame = pd.DataFrame(Tuning_Results)
Tuning_Results_DataFrame.to_csv(os.path.join(Results_3,"Hyperparameter_Tuning_Results.csv"),index=False)

print(Tuning_Results_DataFrame)

print("\nHyperparameter tuning results:")
print(Tuning_Results_DataFrame)

print("\nBest configuration:")
print(Best_Configuration)

print("\nBest validation loss:")
print(Best_Validation_Loss)

print("\nBest model saved as:")
print(Best_Model_Path)
print("Part 4.3 done")

Baseline_Model_Path = os.path.join(Results_3,"Baseline_CNN_Model.keras")
Optimized_Model_Path = os.path.join(Results_3,"Best_Optimized_CNN_Model.keras")
Loaded_Baseline_Model = tf.keras.models.load_model(Baseline_Model_Path)
Loaded_Optimized_Model = tf.keras.models.load_model(Optimized_Model_Path)

print("\nBASELINE MODEL:")
Loaded_Baseline_Model.summary()

print("\nOPTIMIZED MODEL:")
Loaded_Optimized_Model.summary()

Batch_Size = Best_Configuration["Batch_Size"]
Test_Dataset = Create_Dataset(Test_DataFrame,Shuffle=False)

Baseline_Test_Loss, Baseline_Test_Accuracy = (Loaded_Baseline_Model.evaluate(Test_Dataset,verbose=0))
Optimized_Test_Loss, Optimized_Test_Accuracy = (Loaded_Optimized_Model.evaluate(Test_Dataset,verbose=0))

print("\nBaseline model:")
print("Test loss:", Baseline_Test_Loss)
print("Test accuracy:", Baseline_Test_Accuracy)

print("\nOptimized model:")
print("Test loss:", Optimized_Test_Loss)
print("Test accuracy:", Optimized_Test_Accuracy)

# %% PART 5.1

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

Baseline_Model_Path = os.path.join(Results_3,"Baseline_CNN_Model.keras")
Optimized_Model_Path = os.path.join(Results_3,"Best_Optimized_CNN_Model.keras")

Baseline_Model = tf.keras.models.load_model(Baseline_Model_Path)
Optimized_Model = tf.keras.models.load_model(Optimized_Model_Path)

print("Part 5.1 done")


# %% PART 5.2

True_Labels = np.concatenate([Labels.numpy()for Images, Labels in Test_Dataset])

Baseline_Probabilities = Baseline_Model.predict(Test_Dataset)
Optimized_Probabilities = Optimized_Model.predict(Test_Dataset)

Baseline_Predictions = np.argmax(Baseline_Probabilities,axis=1)
Optimized_Predictions = np.argmax(Optimized_Probabilities,axis=1)


Baseline_Accuracy = accuracy_score(True_Labels,Baseline_Predictions)
Baseline_Precision = precision_score(True_Labels,Baseline_Predictions,average="weighted",zero_division=0)
Baseline_Recall = recall_score(True_Labels,Baseline_Predictions,average="weighted",zero_division=0)
Baseline_F1_Score = f1_score(True_Labels,Baseline_Predictions,average="weighted",zero_division=0)

Optimized_Accuracy = accuracy_score(True_Labels,Optimized_Predictions)
Optimized_Precision = precision_score(True_Labels,Optimized_Predictions,average="weighted",zero_division=0)
Optimized_Recall = recall_score(True_Labels,Optimized_Predictions,average="weighted",zero_division=0)
Optimized_F1_Score = f1_score(True_Labels,Optimized_Predictions,average="weighted",zero_division=0)


print("\nBASELINE MODEL RESULTS")

print("Accuracy:", Baseline_Accuracy)
print("Precision:", Baseline_Precision)
print("Recall:", Baseline_Recall)
print("F1-Score:", Baseline_F1_Score)

print("\nBaseline classification report:")

print(classification_report(True_Labels,Baseline_Predictions,target_names=Class_Names,zero_division=0))

print("\nOPTIMIZED MODEL RESULTS")

print("Accuracy:", Optimized_Accuracy)
print("Precision:", Optimized_Precision)
print("Recall:", Optimized_Recall)
print("F1-Score:", Optimized_F1_Score)

print("\nOptimized classification report:")

print(classification_report(True_Labels,Optimized_Predictions,target_names=Class_Names,zero_division=0))


Comparison_DataFrame = pd.DataFrame({
    "Model": ["Baseline CNN","Optimized CNN"],
    "Accuracy": [Baseline_Accuracy,Optimized_Accuracy],
    "Precision": [Baseline_Precision,Optimized_Precision],
    "Recall": [Baseline_Recall,Optimized_Recall],
    "F1_Score": [Baseline_F1_Score,Optimized_F1_Score]})

Comparison_DataFrame.to_csv(os.path.join(Results_3,"Baseline_Optimized_Comparison.csv"),index=False)

print("\nModel comparison:")
print(Comparison_DataFrame)

print("Part 5.2 done")


# %% PART 5.3

Optimized_History_DataFrame = pd.read_csv(os.path.join(Results_3,"Best_Optimized_Model_History.csv"))

Baseline_Epochs = range(1,len(History.history["loss"]) + 1)
Optimized_Epochs = range(1,len(Optimized_History_DataFrame) + 1)
Figure = plt.figure(figsize=(18, 10),constrained_layout=True)
Grid = Figure.add_gridspec(2,3)

Axis_1 = Figure.add_subplot(Grid[0, 0])
Axis_1.plot(Baseline_Epochs,History.history["loss"],label="Training")
Axis_1.plot(Baseline_Epochs,History.history["val_loss"],label="Validation")
Axis_1.set_title("Baseline CNN Loss")
Axis_1.set_xlabel("Epoch")
Axis_1.set_ylabel("Cross-Entropy Loss")
Axis_1.legend()
Axis_1.grid(alpha=0.3)

Axis_2 = Figure.add_subplot(Grid[0, 1])
Axis_2.plot(Baseline_Epochs,History.history["accuracy"],label="Training")
Axis_2.plot(Baseline_Epochs,History.history["val_accuracy"],label="Validation")
Axis_2.set_title("Baseline CNN Accuracy")
Axis_2.set_xlabel("Epoch")
Axis_2.set_ylabel("Accuracy")
Axis_2.set_ylim(0,1)
Axis_2.legend()
Axis_2.grid(alpha=0.3)

Axis_3 = Figure.add_subplot(Grid[1, 0])
Axis_3.plot(Optimized_Epochs,Optimized_History_DataFrame["loss"],label="Training")
Axis_3.plot(Optimized_Epochs,Optimized_History_DataFrame["val_loss"],label="Validation")
Axis_3.set_title("Optimized CNN Loss")
Axis_3.set_xlabel("Epoch")
Axis_3.set_ylabel("Cross-Entropy Loss")
Axis_3.legend()
Axis_3.grid(alpha=0.3)

Axis_4 = Figure.add_subplot(Grid[1, 1])
Axis_4.plot(Optimized_Epochs,Optimized_History_DataFrame["accuracy"],label="Training")
Axis_4.plot(Optimized_Epochs,Optimized_History_DataFrame["val_accuracy"],label="Validation")
Axis_4.set_title("Optimized CNN Accuracy")
Axis_4.set_xlabel("Epoch")
Axis_4.set_ylabel("Accuracy")
Axis_4.set_ylim(0,1)
Axis_4.legend()
Axis_4.grid(alpha=0.3)

Axis_5 = Figure.add_subplot(Grid[:, 2])
Optimized_Confusion_Matrix = confusion_matrix(True_Labels,Optimized_Predictions)
Confusion_Matrix_Display = ConfusionMatrixDisplay(confusion_matrix=Optimized_Confusion_Matrix,display_labels=Class_Names)
Confusion_Matrix_Display.plot(ax=Axis_5,cmap="Blues",colorbar=False,values_format="d")
Axis_5.set_title("Optimized CNN Confusion Matrix")
Axis_5.tick_params(axis="x",labelrotation=45)

Figure.suptitle("Baseline and Optimized CNN Evaluation",fontsize=16)
Figure.savefig(os.path.join(Results_3,"Baseline_Optimized_Evaluation.png"),dpi=150)
plt.show()

print("Part 5.3 done")
