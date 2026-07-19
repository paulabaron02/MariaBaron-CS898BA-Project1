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

To find the best CNN configuration for the fish classification task, a grid search was performed by testing different combinations of three hyperparameters: learning rate (0.01, 0.001, and 0.0001), batch size (32 and 64), and dropout rate (0.3 and 0.5). This resulted in a total of 12 different configurations. For each combination, a new CNN model was built, compiled using the Adam optimizer, and trained using the training and validation datasets. Before each training session, the Keras backend session was cleared to ensure that every experiment started independently.

During training, the validation loss was monitored to identify the best-performing epoch for each configuration. The minimum validation loss, the corresponding validation accuracy, and the best epoch were recorded. Whenever a configuration achieved better validation performance than the previous ones, its model was saved as Best_Optimized_CNN_Model.keras.

After all configurations were evaluated, the results were stored in the file Hyperparameter_Tuning_Results for comparison. Finally, both the baseline CNN and the optimized CNN were evaluated using the test dataset to compare their loss and accuracy and determine how much performance improved after hyperparameter tuning.

---

## Part 5: Evaluation and Analysis

**Part 5.1** 

The baseline CNN and the optimized CNN were evaluated using the same held-out test dataset to analyze the effect of data augmentation and hyperparameter tuning on the model performance.

Data augmentation (horizontal flip, random rotation, and random brightness) helped increase the diversity of the training images and reduced overfitting during training. Both models showed a continuous decrease in training loss while maintaining relatively stable validation accuracy, indicating that the augmentation techniques improved the model's ability to generalize.

During hyperparameter tuning, twelve different combinations of learning rate, batch size, and dropout rate were evaluated. The best validation performance was obtained using a learning rate of 0.001, a batch size of 64, and a dropout rate of 0.3. Although this configuration produced the lowest validation loss among the tested combinations, it did not outperform the baseline model on the independent test dataset.

This result suggests that the selected hyperparameters improved optimization during validation but did not provide better generalization to unseen data. Therefore, the baseline architecture remained the best-performing model for this dataset.

**Part 5.2**

The baseline CNN achieved a test accuracy of 81.70%, while the optimized CNN obtained 79.74%. Similar behavior was observed for the remaining evaluation metrics.

| Metric |	Baseline CNN |	Optimized CNN |
|---|---|---|
| Accuracy |	81.70%	| 79.74% |
| Precision	| 81.35%	| 80.47% |
| Recall	| 81.70%	| 79.74% |
| F1-Score	| 80.88%	| 79.78% |

The baseline model achieved higher overall performance across all evaluation metrics.
From the class-wise classification reports, both models performed well on the Discuss and Gold classes, while the Cray and Oscar classes remained the most challenging. These classes showed lower recall and F1-scores, indicating that they were more frequently confused with other fish species.
Although the optimized model slightly improved precision for the Discuss class (1.00 compared to 0.94), this improvement was not sufficient to increase the overall classification performance.
Overall, the quantitative evaluation indicates that hyperparameter tuning did not improve the final classification accuracy for this dataset.

**Part 5.3**

The training and validation curves show that both models successfully learned the classification task. Training loss decreased steadily throughout the training process, while validation accuracy remained relatively stable.

<img width="2700" height="1500" alt="Baseline_Optimized_Evaluation" src="https://github.com/user-attachments/assets/385bc2ea-cf4e-4b80-a5cb-574fee05276b" />

The optimized CNN exhibited slightly smoother learning curves due to the inclusion of dropout regularization. However, this additional regularization also reduced the final test accuracy, suggesting that the model became slightly underfitted for this relatively small dataset.

The confusion matrix confirms that most prediction errors occurred between visually similar fish species. The largest number of misclassifications involved the Oscar and Cray classes, whereas the Discuss, Gold, and Guppy classes were classified with higher consistency.
Overall, the visual analysis agrees with the quantitative metrics, showing that the baseline CNN generalized slightly better than the optimized CNN despite the hyperparameter tuning process.

---

## Part 6: Submission

**Submit your updated GitHub repository link on Blackboard.** Ensure your master/main branch or pull request clearly reflects the additions made for Homework Three.
