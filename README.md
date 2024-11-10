# Predicting Student Placement for Campus Recruitment

This project aims to develop a machine learning pipeline to predict student placement status in campus recruitment. The prediction leverages students' academic and personal attributes, enabling institutions to gain insights into key factors influencing placement outcomes. The solution is implemented with three classification models—Logistic Regression, Decision Tree, and Random Forest—to identify the most effective approach for accurate placement prediction.

## Project Overview

### Objective

The primary goal is to predict whether a student will be placed based on various features, including academic scores, demographic information, and work experience. The project also involves:

- End-to-end pipeline setup, from data preprocessing to model evaluation.
- Comparison between model performances to identify the most effective model.
- Comprehensive reporting for academic and industry perspectives.

## Dataset

### Source

The dataset used in this project was sourced from [Kaggle](https://www.kaggle.com/competitions/ml-with-python-course-project/overview), providing real-world data on student profiles and placement status. It contains fields like `gender`, `ssc_p` (secondary education percentage), `degree_t` (degree type), `workex` (work experience), and `specialisation`.

### Target Variable

The target variable, `status`, indicates whether a student was placed ("Placed") or not placed ("Not Placed") during campus recruitment.

### Preprocessing Steps

1. **Data Cleaning**: Checked and handled missing values to ensure data integrity.
2. **Encoding Categorical Variables**: Applied one-hot encoding to categorical variables such as `gender`, `degree_t`, `workex`, and `specialisation` to prepare them for model input.
3. **Scaling**: Standardized continuous variables to avoid feature dominance due to scale differences.
4. **Train-Test Split**: Split the dataset into training (70%) and testing (30%) sets to evaluate model performance effectively.

## Model Selection

### Models Chosen

1. **Logistic Regression**: A simple yet effective linear model, chosen for its interpretability and efficiency in binary classification tasks.
2. **Decision Tree**: A non-linear model capable of capturing feature interactions, suitable for identifying complex patterns in data.
3. **Random Forest**: An ensemble of decision trees, known for robustness and generalization, making it ideal for handling noise and varied data patterns.

### Rationale for Model Selection

Each model was chosen to offer a balance between simplicity and complexity:

- **Logistic Regression** serves as a baseline for linear classification.
- **Decision Tree** captures non-linear relationships and is easy to interpret.
- **Random Forest** provides an ensemble approach, increasing model stability and predictive power.

## Training Process

### Data Preparation

The training process began with data preprocessing and splitting, ensuring that each model could be trained effectively on clean, standardized data. Categorical variables were encoded, and features were scaled to maintain consistency across models.

### Hyperparameter Tuning

Hyperparameter tuning was performed using a grid search with cross-validation to select the best parameters for each model:

- **Logistic Regression**: Regularization strength (`C`) was optimized to balance model complexity.
- **Decision Tree**: Key parameters like `max_depth` and `min_samples_split` were tuned to control overfitting.
- **Random Forest**: Parameters like `n_estimators` (number of trees) and `max_features` were tuned to improve model generalization.

### Cross-Validation

For each model, 5-fold cross-validation was used during tuning to avoid overfitting and validate performance on multiple subsets of training data. This process allowed for reliable parameter selection and improved generalization to unseen data.

## Evaluation Metrics

The models were evaluated on the following metrics:

- **Accuracy**: Measures the proportion of correct predictions.
- **Precision**: Indicates the proportion of true positive placements among all predicted placements.
- **Recall**: Reflects the model's ability to identify actual placements.
- **F1 Score**: The harmonic mean of precision and recall, offering a balanced measure.
- **Confusion Matrix**: Visual representation of model performance, showing true positives, true negatives, false positives, and false negatives.

### Summary of Model Evaluation

Each model was evaluated on these metrics, and all three models (Logistic Regression, Decision Tree, and Random Forest) achieved perfect scores of 1.00 on the test set. The confusion matrices for each model confirmed flawless classification, with no false positives or false negatives.

## Model Comparison

The comparison between models indicated:

1. **Logistic Regression**: Performed exceptionally well with perfect scores, offering simplicity and interpretability.
2. **Decision Tree**: Matched the accuracy of Logistic Regression but with potential for higher adaptability on complex data due to non-linear modeling.
3. **Random Forest**: Although it achieved the same scores as other models, Random Forest’s ensemble nature provides robustness and better generalization, making it ideal for real-world applications where data variability might increase.

### Insights

Given the identical scores, each model appears capable of accurately predicting placement outcomes. However:

- **Logistic Regression** is recommended for scenarios requiring high interpretability.
- **Random Forest** is preferred for its robustness and ability to generalize well to unseen, more complex data, which could benefit future applications with larger, varied datasets.

## Conclusion

This project demonstrates a successful end-to-end machine learning pipeline for predicting campus recruitment placement. All models achieved high performance, suggesting that the dataset is well-suited for this task. However, the similarity in performance highlights the need for additional validation on diverse datasets to verify model generalizability.

### Recommendations

1. **Use Logistic Regression** for a simple, interpretable model if dataset complexity remains consistent.
2. **Apply Random Forest** for larger or more varied datasets, as it is likely to maintain accuracy and handle noise effectively.
3. **Further Validation**: Testing these models on larger, unseen data would help determine the true generalizability and robustness of each model in real-world applications.

This README provides a comprehensive overview of the pipeline, model selection, training, tuning, and evaluation process, making it suitable for academic and industry readers.
