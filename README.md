# Credit Risk Modeling using Deep Neural Networks

An end-to-end **credit risk modeling** project built with **Python** and **Deep Learning** .

**Dataset is already preprocessed** 

This project demonstrates how model performance can be systematically improved through
feature engineering, categorical embeddings, hyperparameter optimization, regularization,
and Wide & Deep architectures.

==> Project Focused On **improve deep model performance with tunning , wide&deep and regularixation** <==

## Dataset

- **Default of Credit Card Clients Dataset**
- Source: https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset

rows:
- `ID`: Client identifier
- `LIMIT_BAL`: Amount of given credit
- `SEX`: Gender
- `EDUCATION`: Education level
- `MARRIAGE`: Marital status
- `AGE`: Age
- `PAY_0` – `PAY_6`: Repayment status for the past 6 months
- `BILL_AMT1` – `BILL_AMT6`: Bill statement amounts
- `PAY_AMT1` – `PAY_AMT6`: Previous payment amounts
- `default.payment.next.month`: Target variable

## Project Pipeline

1_Data Preparation
        ↓
2_Baseline Neural Network Model
        ↓
3_Feature Scaling and Encoding
        ↓
4_Categorical Embedding-Based Model
        ↓
5_Hyperparameter Optimization (Keras Tuner)
        ↓
6_Regularization Analysis (Dropout & Weight Decay) 
        ↓
7_Wide & Deep Neural Network Architecture
        ↓
8_Threshold Optimization using Precision–Recall Trade-off
        ↓
9_Model Evaluation (Confusion Matrix, ROC AUC, Classification Report)



### 🔹 1_Data Preparation

The dataset is already preprocessed.
At this stage, I only prepare the features and split the data into training, validation, and test sets.

### 🔹 2_Baseline Neural Network Model

First, I train a simple neural network on the raw data without normalization or encoding.

Results:
=> val_accuracy: 0.6876
=> val_loss: 150.7564

The model achieves about 68% accuracy, but the training is unstable and noisy (validation metrics fluctuate significantly).

### 🔹 3_Feature Scaling and Encoding

Next, I normalize numerical features and encode categorical columns using One-Hot Encoding, then retrain the model.

Results:
=> val_accuracy: 0.7789
=> val_loss: 286.8295

Accuracy improves to 77% , which is a significant improvement over the baseline model.

### 🔹 4_Categorical Embedding-Based Model

Instead of One-Hot Encoding, I use embedding layers for categorical features and train the model again.

Results:
=> val_accuracy: 0.806
=> val_loss: 0.443

Using embeddings for only three categorical columns improves accuracy by about 3%, while also stabilizing training.

### 🔹 5_Hyperparameter Optimization

I apply Hyperband to tune the model’s hyperparameters.

Tuned parameters:
- **Number of hidden layers** 
- **Number of neurons per layer** 
- **Dropout rate**
- **L2 regularization strength** 
- **Optimizer**
- **Activation function**

=> Best val_accuracy So Far: 0.821373

After tuning, the model reaches approximately 82% validation accuracy.

### 🔹 6_Regularization Analysis (Dropout & Weight Decay) 

Using the best hyperparameters, I redesign the model and experiment with different Dropout and L2 regularization values.
By analyzing training and validation loss curves, I select the regularization settings that best control overfitting.

### 🔹 7_Wide & Deep Neural Network Architecture

To capture both memorization and generalization, I implement a Wide & Deep architecture, allowing the model to learn from raw features and deep representations simultaneously.

### 🔹 8_Threshold Optimization

Since this is a credit risk problem, false positives are costly.
I adjust the classification threshold to achieve ~74% precision.

For a more conservative banking strategy, the threshold can be shifted to prioritize recall instead.

### 🔹 9_Model Evaluation

Finally, I evaluate the model on a held-out test set that was never seen during training or tuning.

Final results:
=> Test Accuracy: 82.48%

Overall, the model demonstrates strong and stable performance.

## 👤 Author

**Shervin**  
Computer Science Student  
Focused on **Machine Learning** and **Data Science** 
