# Predictive-model-for-Industry-Machinery


### Introduction

The objective of this project is to predict Failure_Maintenance_Indicator using machine learning models based on the provided dataset received as part of our project under Entri organization. This involves exploring various models, tuning their hyperparameters, and evaluating their performance to identify the best model.

### Data Description
The dataset consists of 1000 rows and 17 columns, including features such as:

Equipment_ID
Sensor_1
Sensor_2
Sensor_3
Environmental_Temperature
Environmental_Humidity
Production_Volume
Operating_Hours
Error_Code
Equipment_Age
Power_Consumption
Voltage_Fluctuations
Current_Fluctuations
Vibration_Analysis
Temperature_Gradients
Pressure_Levels
Target variable: Failure_Maintenance_Indicator

### Preprocessing

#### Data preprocessing involved:

Absence of missing values
Feature selection including SelectKBest,Recursive Feature Elimination,Random Forest & Chi Square. After exploring all the feature selection techniques conluded SelectKBest stands out as the most effective method
Scaling the data using MinMax Scaler


### Modeling

The following models were used:

Logistic Regression
Decision Tree
Random Forest
Support Vector Classifier
K-Nearest Neighbors
Multi-Layer Perceptron
Gradient Boosting
Naive Bayes

Hyperparameters were tuned using GridSearchCV with specific grids for each model.


### Evaluation
Evaluation Metrics:

Accuracy: Overall accuracy of the model.
Confusion Matrix: Breakdown of true positives, true negatives, false positives, and false negatives.
Classification Report: Precision, recall, F1-score for each class.
ROC AUC Score: Area under the ROC curve.
Specificity: Proportion of true negatives.

The evaluation metrics for each model are as follows:
##### Logistic Regression:
Accuracy: 0.5050
Specificity: 1.000000
ROC AUC Score:0.513024
Mean CV Score:0.510750	

##### Decision Tree Classifier:
Accuracy: 0.5100
Specificity: 0.366337
ROC AUC Score:0.515340
Mean CV Score:0.499125	

##### Random Forest Classifier:
Accuracy:0.5005
Specificity: 0.560396
ROC AUC Score:0.499328
Mean CV Score:0.493750	


##### K-Nearest Neighbors:
Accuracy:0.4860
Specificity:0.505941
ROC AUC Score:0.489616
Mean CV Score:0.500000

##### Naive Bayes:
Accuracy:0.5240
Specificity:0.7089
ROC AUC Score:0.517923
Mean CV Score:0.509125


##### Support Vector Classifier:
Accuracy:0.5040
Specificity:0.7950
ROC AUC Score:0.51460
Mean CV Score:0.510750

##### Gradient Boosting:
Accuracy:0.5040
Specificity:0.7950
ROC AUC Score:0.508986
Mean CV Score:0.509250

##### MLP Classifier:
Accuracy:0.5010
Specificity:0.631683
ROC AUC Score:0.514430
Mean CV Score:0.498375


### Conclusion

The best performing model is Naive Bayes with an accuracy of 0.52, specificity of 0.70, and ROC AUC of 0.51 We recommend 
- Collecting more data to enhance model training.
- Exploring additional feature engineering techniques.
- Conducting more in-depth hyperparameter tuning for even better performance.








### Classification Algorithms: A Comprehensive Overview

In machine learning, classification algorithms play a crucial role in predicting categorical outcomes based on input features. These algorithms analyze labeled training data to learn patterns and relationships between input variables and their corresponding classes. Here, we'll explore seven popular classification algorithms in detail.

#### 1. Logistic Regression:
###### Overview: Logistic regression is a widely-used linear classification algorithm that predicts the probability of an instance belonging to a particular class.
###### How it Works: It models the relationship between the dependent variable (target) and one or more independent variables (features) by estimating probabilities using a logistic function.
###### Pros: Simple, interpretable, efficient for binary classification tasks.
###### Cons: Limited to linear decision boundaries, may not perform well with complex data.

#### 2. Decision Tree Classifier:
###### Overview: Decision tree classifier is a non-parametric supervised learning algorithm that partitions the feature space into segments based on the value of input features.
###### How it Works: It builds a tree-like structure where each internal node represents a decision based on a feature attribute, and each leaf node represents a class label.
###### Pros: Intuitive, easy to interpret, handles both numerical and categorical data.
###### Cons: Prone to overfitting, sensitive to small variations in the data.

#### 3. Random Forest Classifier:
###### Overview: Random forest classifier is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes (classification) or mean prediction (regression) of the individual trees.
###### How it Works: It builds each tree on a random subset of features and combines predictions through voting or averaging.
###### Pros: Robust against overfitting, handles high-dimensional data, provides feature importance ranking.
###### Cons: Requires more computational resources, less interpretable compared to single decision trees.

#### 4. K-Nearest Neighbors (KNN):
###### Overview: KNN is a simple and intuitive classification algorithm that predicts the class of a new data point based on the majority class of its k nearest neighbors in the feature space.
###### How it Works: It computes the distance between the query instance and all the training samples to find the k nearest neighbors, then assigns the class label based on the majority vote.
###### Pros: Easy to understand, does not require training, suitable for small datasets.
###### Cons: Computationally expensive for large datasets, sensitive to irrelevant features and noise.

#### 5. Naive Bayes:
###### Overview: Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem with an assumption of independence between features.
###### How it Works: It calculates the probability of each class given the input features and selects the class with the highest probability.
###### Pros: Simple, fast, works well with high-dimensional data, robust to irrelevant features.
###### Cons: Assumes independence between features (which may not hold true), may suffer from the zero-frequency problem.

#### 6. Support Vector Classifier (SVC):
###### Overview: Support Vector Classifier (SVC) is a powerful classification algorithm that constructs hyperplanes in a high-dimensional space to separate instances of different classes.
###### How it Works: It aims to find the optimal hyperplane that maximizes the margin between classes while minimizing classification errors.
###### Pros: Effective in high-dimensional spaces, versatile (kernel functions allow nonlinear decision boundaries), robust to overfitting.
###### Cons: Computationally intensive, sensitive to the choice of kernel and regularization parameters.

#### 7. Gradient Boosting:
###### Overview: Gradient Boosting is an ensemble learning technique that builds a strong predictive model by combining the predictions of multiple weak learners (typically decision trees) sequentially.
###### How it Works: It fits a series of decision trees to the residuals of the preceding trees, gradually reducing the error in prediction.
###### Pros: High predictive accuracy, handles heterogeneous data types, automatically handles missing values.
###### Cons: Prone to overfitting if the number of trees is too large, sensitive to hyperparameters tuning.


In summary, each classification algorithm has its strengths and weaknesses, and the choice of algorithm depends on various factors such as the nature of the data, computational resources, interpretability requirements, and performance metrics. It's essential to experiment with different algorithms and fine-tune their parameters to achieve the best results for a given classification task.
