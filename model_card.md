# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This project uses a machine learning classification model to predict whether a person’s income is greater than 50K or less than or equal to 50K. The model was built with the Census Income dataset provided in the project starter files.
The model uses a Random Forest classifier from scikit-learn. A Random Forest is a supervised learning algorithm that combines many decision trees to make a final prediction. This model was chosen because it works well for classification problems and can handle a mix of numerical and encoded categorical features.
Before training, the categorical features were converted into numeric values using one-hot encoding. The salary label was converted into a binary target so the model could learn the difference between the two income classes.


## Intended Use
The intended use of this model is to demonstrate a complete machine learning pipeline for WGU's D501 course project 2. The pipeline includes data loading, preprocessing, model training, model evaluation, model serialization, slice performance evaluation, unit testing, and API inference using FastAPI.
This model is for educational purposes only.


## Training Data
The training data comes from the Census Income dataset included in the project repository. The dataset includes information such as age, workclass, education, marital status, occupation, relationship, race, sex, capital gain, capital loss, hours per week, native country, and salary.
The dataset was split into a training set and a test set using an 80/20 train-test split. The training set contained 26,048 rows and 15 columns before preprocessing. The categorical columns were one-hot encoded, and the salary column was used as the target label.


## Evaluation Data
The evaluation data came from the held-out test set. The test set contained 6,513 rows and 15 columns before preprocessing. The test data was processed using the same encoder that was fitted on the training data. This helps make sure the model receives the same type of input during evaluation that it received during training. The test set was used to measure how well the trained model performed on data it had not seen during training.

## Metrics
The model was evaluated using precision, recall, and F1 score. Precision measures how many positive predictions were correct. Recall measures how many actual positive cases were correctly identified. The F1 score balances precision and recall into one metric.
On the test dataset, the model achieved the following performance:
•	Precision: 0.7419
•	Recall: 0.6384
•	F1 score: 0.6863
These results show that the model can make useful predictions, but it is not perfect. The recall score is lower than the precision score, which means the model misses some positive cases. 
The model also produced slice performance results for categorical features. These results were saved in slice_output.txt.


## Ethical Considerations
This dataset includes sensitive demographic information such as race, sex, relationship status, and native country. Because of this, the model may learn patterns that reflect bias or inequality in the original data. Even if the model performs reasonably well overall, it may perform differently for different demographic groups.
For this reason, this model should not be used for real-world decisions that affect people. The slice performance file is included to help review how model performance changes across different categorical values, but more fairness testing would be needed before considering any real use.


## Caveats and Recommendations
This model was created for a machine learning DevOps course project and is not intended for production use. The model was trained on a historical census dataset, so its predictions may not generalize well to modern populations or different data sources.
Future improvements could include additional fairness analysis, hyperparameter tuning, cross-validation, feature importance review, and monitoring for model drift. Before any real-world use, the model should be evaluated carefully for bias, fairness, privacy concerns, and performance across demographic subgroups.
