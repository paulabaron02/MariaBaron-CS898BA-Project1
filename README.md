# Homework Three: Deep Learning for Fish Classification

CS 898BA – Image Analysis and Computer Vision

Maria Paula Baron Rodriguez

**Wsu ID:** J858Q278

**Purpose:** To design, train, and optimize a Convolutional Neural Network (CNN) for image classification and evaluate hyperparameter tuning strategies.

This script is the same one the professor provided for the assignment, but modified to display the results, the setup, and the execution of the steps.

## Situation
Your supervisor’s supervisor has returned from his meeting, but his focus has completely shifted. The "alien" bio-mass calculations were inconclusive, and the department has suddenly secured a grant to monitor local marine ecosystems. He drops a dataset of fish images on your desk, muttering about "automating species identification to track ecological diversity," before rushing off to a faculty luncheon. 

Armed with your deep learning pipeline knowledge, you decide to construct a robust convolutional neural network to classify the fish species and systematically tune its hyperparameters to maximize performance.

---

## Part 1: Repository Maintenance & Logistics

1. **Branching:** Do not overwrite your Homework Two code. Create a new branch in your existing repository named `Feature-Classification`.
2. **AI Log:** Continue tracking all AI usage in your `AI_Log.md` file following the exact same format as previous assignments.
3. **Commit Strategy:** Perform incremental commits with clear, professional messages as you build, train, and optimize your network.

---

## Part 2: Data Preprocessing & Augmentation

**Part 2.1** 

The code iterates through the subfolders of the Fish directory, extracts the paths of all images along with their respective labels, and constructs a DataFrame. It then performs a stratified split to divide the data into 70% Training, 15% Validation, and 15% Testing, ensuring that all classes maintain the same proportion within each subset. This organization allows the model to be trained, validated, and finally evaluated on unseen data. According to Mastering OpenCV 4 with Python, a proper separation of the dataset helps reduce overfitting and provides a more reliable evaluation of the model's ability to generalize to new images.

**Part 2.2** 

tf.data pipeline is created to resize all images to 128 × 128 pixels, convert them to the float32 data type, and normalize their intensity values to the [0,1] range by dividing by 255. The pipeline also uses batching, parallel loading, and prefetching to efficiently prepare the data for training. As described in Mastering OpenCV 4 with Python, images are represented as multidimensional arrays of pixel values, so preprocessing them into a consistent size and numerical scale ensures they can be processed efficiently by deep learning models.

**Part 2.3**

A sequential data augmentation block is created to apply random horizontal flips, rotations, and brightness adjustments during training. These transformations increase the variability of the training images without changing their labels, making the model more robust to changes in orientation and lighting. As explained in Mastering OpenCV 4 with Python, deep learning models generally achieve better performance when trained with larger and more diverse datasets, making data augmentation an effective technique for improving generalization and reducing overfitting.

---

## Part 3:

**Part 3.1**

For this part a neural network is created to classify the fish images. The model receives RGB images of 128×128 pixels and first applies the previously defined data augmentation transformations. It then uses three Conv2D and MaxPooling2D blocks with 32, 64, and 128 filters. The convolutional layers progressively learn visual features such as edges, textures, colors, and fish shapes, while the pooling layers reduce the spatial dimensions and computational cost. This follows the deep-learning principle described in Mastering OpenCV 4 with Python, where multiple layers progressively transform the input and automatically extract increasingly complex features for classification.

After feature extraction, the Flatten layer converts the feature maps into a one-dimensional vector. A dense layer with 128 neurons combines the learned features, while the final softmax layer produces a probability for each fish class. The class with the highest probability is selected as the model’s prediction.

**Part 3.2**

In this part, the model is compiled using the Adam optimizer with a learning rate of 0.001 and the sparse_categorical_crossentropy loss function because the fish labels are represented as integer indices. The model is then trained for 10 epochs using the training dataset, while its performance is evaluated with the validation dataset after each epoch.

According to the graphicson the folder Results 3 the model showed consistent learning throughout training, with decreasing loss and increasing accuracy. It achieved training accuracy and validation accuracy. The difference between the training and validation curves indicates slight overfitting during the final epochs, although the model still demonstrated good classification performance.

---

## Part 4: Hyperparameter Optimization

To push the classification metrics higher, perform a systematic hyperparameter tuning experiment:

1. **Tuning Strategy:**
   * Select a tuning strategy (e.g., Grid Search, Random Search, or Bayesian Optimization via Optuna/KerasTuner).
   * Experiment with a minimum of three distinct hyperparameters:
     * **Learning Rate:** Test at least 3 values (e.g., 0.01, 0.001, 0.0001).
     * **Batch Size:** Test at least 2 configurations (e.g., 32, 64).
     * **Regularization:** Introduce and vary Dropout rates (e.g., 0.3 vs. 0.5) or Weight Decay ($L_2$ regularization) to combat overfitting.

> **Note:** Identify and save the best-performing model configuration based on validation loss.

---

## Part 5: Evaluation and Analysis

Evaluate your baseline model alongside your optimized model using the held-out test dataset.

1. **Qulitative Analysis:**
   * Discuss how specific data augmentation techniques impacted training stability. Analyze the effects of your hyperparameter adjustments, detailing which parameters yielded the most significant improvements in preventing overfitting or accelerating convergence.
2. **Quantitative Comparison:**
   * Generate a comprehensive classification report for both the baseline and optimized models. Calculate and print the **Accuracy, Precision, Recall, and F1-Score** across all fish classes.

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

3. **Visualization:**
   * Generate a side-by-side visualization including the training vs. validation loss/accuracy curves for both models, alongside a multi-class **Confusion Matrix** of the optimized model's test performance. Include this image grid in your updated `README.md`.

---

## Part 6: Submission

**Submit your updated GitHub repository link on Blackboard.** Ensure your master/main branch or pull request clearly reflects the additions made for Homework Three.
