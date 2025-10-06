# Machine Learning Fundamentals

## 1. Core ML Concepts
1. What is machine learning, and how does it differ from traditional programming?
    
    `Machine Learning it is a field of Artificial Intelligence that uses algorithms and statistical models to enable computer systems to learn patterns from data, and make predictions or decisions without being explicitly programmed. Unlike traditional programming, where rules each rule manually codded, ML models learn rules automatically from data.`

2. What’s the difference between supervised and unsupervised learning?
    
    `Supervised leaning uses labeled data to train a model, where correct answers is knows. Unsupervised learning uses unlabeled data to find pattern or structure in the data, such as clustering and dimensionality reduction.`

3. Give an example of a regression and a classification problem.
    
    `Regression predicts a continuous value ie house price. Classification predict categorical value, ie span or not`

4. Why do we split data into training, validation, and test sets?

    `We split dataset into three sets to train the model, tune hyperparameters, and evaluate performance on unseen data. This helps to ensure that model learns pattern rather than just memorizing training data.`

5. Explain bias and variance. How do they trade off?
    
    `Bias is the error due to a model being too simple to capture the underlying patterns in the data, leading to underfitting. Variance is the error due to a model being too complex and sensitive to training data, leading to overfitting. The bias-variance trade-off describes how decrease in one often increases the other, and the goal is to find a balance for good generalization.`

6. What are the main causes of overfitting, and how can you prevent it?

    `Overfitting occurs when a model learns a noise or irrelevant patterns in the training data. It can be prevented by using simpler models, reducing number of features, applying regularization, increasing training data, and using early stopping during training.`

7. What is cross-validation and why is it important?

    `Cross-validation is a validation technique to evaluate a model's ability to generalize on unseen data by repeatedly splitting the dataset in to training and validation subsets. It is important because we want to assess model performance on data that was not used for training.`

8. What’s the difference between k-fold and stratified k-fold validation?

    `K-fold cross-validation splits the dataset into k equal subsets (folds). For each fold, the different test subset is used. This allows for a more reliable estimate of model performance. Stratified K-fold is similar but ensures that the proportion of each class is maintained in every fold, which is important for imbalanced classification dataset.`

9. Explain the concept of “data leakage” and how to avoid it.

    `This happens when model is trained on the data that wouldn't be available in production usage, leading the inaccurate performance and wrong insights.`

10. What are some common evaluation metrics for classification and regression?

    `Regression models: r**2, Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE). Classification models evaluated using confusion matrix: True Positives, True Negatives, False Positives and False Negatives. Accuracy: how many is correct; Precision: positive predictions are correct; Recall/Sensitivity; F1 Score; ROC Curve.`


## 2. Data Preprocessing & Feature Engineering

### Key Topics
* Data cleaning
* Missing values
* Categorical encoding (One-Hot, Label Encoding)
* Normalization and Standardization
* Feature selection and dimensionality reduction

### Practice Questions

1. How do you handle missing or corrupted data?
2. When should you normalize features vs standardize them?
3. What are the advantages and disadvantages of one-hot encoding?
4. What is PCA and when would you use it?
5. Why might feature scaling be necessary before training certain models?
6. How does feature correlation affect model performance?
7. What is the curse of dimensionality?
8. What are feature importance scores, and how are they computed?

## 3. Model Training & Optimization

### Key Topics

* Loss functions
* Gradient Descent (and variants: SGD, Adam, RMSProp)
* Regularization (L1, L2, ElasticNet)
* Hyperparameter tuning (GridSearch, RandomSearch, Bayesian)

### Practice Questions

1. What is a loss function? Give examples for regression and classification.
2. Explain how gradient descent works.
3. What is the learning rate and what happens if it’s too high or too low?
4. Compare L1 and L2 regularization.
5. What does early stopping do, and why is it useful?
6. How do you tune hyperparameters efficiently?
7. What are the pros/cons of manual tuning vs automated methods?

## 4. Model Evaluation

### Key Topics

* Accuracy, Precision, Recall, F1-score
* Confusion matrix
* ROC Curve & AUC
* R², RMSE, MAE for regression
* Cross-validation and bootstrapping

### Practice Questions

1. What’s the difference between precision and recall?
2. How is the F1-score calculated, and when is it useful?
3. How would you evaluate a model on imbalanced data?
4. What’s the ROC curve and what does AUC measure?
5. How do you interpret R²?
6. Why might accuracy be misleading?

## 5. ML Workflow & Practical Considerations

### Key Topics

* End-to-end ML pipeline
* Train-test drift and data leakage
* Model interpretability
* Scalability & productionization

### Practice Questions

1. Describe the steps in an end-to-end machine learning project.
2. What’s the difference between data drift and concept drift?
3. How can you interpret black-box models?
4. What are the main challenges in deploying ML models to production?
5. How do you monitor model performance over time?

## 6. Probability & Statistics Refresher (Essential for ML)

### Key Topics

* Probability distributions
* Mean, median, variance, standard deviation
* Bayes’ theorem
* Correlation vs causation
* Sampling bias

### Practice Questions

1. What is the difference between variance and standard deviation?
2. Explain Bayes’ theorem with an example.
3. What is the difference between correlation and causation?
4. What is sampling bias and how can it affect model outcomes?
5. Why is understanding probability important in ML?

## 7. Conceptual Coding Challenges (Python / Scikit-Learn)

Try implementing:

1. A simple **train/validation/test split** manually.
2. A **cross-validation** loop using `sklearn.model_selection`.
3. **Feature normalization** using `StandardScaler`.
4. A **logistic regression classifier** from scratch using gradient descent.
5. Compare overfitting between a **decision tree** and **random forest**.
6. Tune hyperparameters using `GridSearchCV`.
7. Visualize performance with a **confusion matrix** and **ROC curve**.
