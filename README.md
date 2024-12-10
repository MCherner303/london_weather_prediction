London Weather Prediction with Machine Learning
Project Overview
This project focuses on building a machine learning pipeline to predict the mean temperature (°C) in London, England, based on various meteorological factors. As climate change impacts weather patterns, accurate temperature predictions are becoming increasingly important for businesses and decision-making. Using data from the london_weather.csv dataset, this project explores different regression models to achieve a Root Mean Squared Error (RMSE) ≤ 3, leveraging MLflow for experiment tracking.

Dataset
The dataset contains daily weather data, including:

Features:
cloud_cover (oktas), sunshine (hours), global_radiation (W/m²), max_temp (°C), min_temp (°C), precipitation (mm), pressure (Pa).
Target:
mean_temp (°C).
Size: 15,341 rows, 10 columns.
Data Preprocessing
Handling Missing Values:
Dropped rows with missing target values (mean_temp).
Imputed missing predictor values (cloud_cover, global_radiation, etc.) using the mean.
Excluded the snow_depth column due to significant missing data.
Feature Engineering:
Converted the date column into a datetime object for potential future analysis (though not directly used in modeling).
Methodology
1. Baseline Model
A Linear Regression model was trained as a baseline, achieving an RMSE of 0.92, demonstrating that the predictors have strong relationships with the target variable.

2. Advanced Models
Two advanced regression models were trained:

Random Forest Regressor: RMSE = 0.92.
Gradient Boosting Regressor: RMSE = 0.90.
Both models were logged and tracked using MLflow, including their hyperparameters and evaluation metrics.

3. Hyperparameter Tuning
Conducted hyperparameter tuning for the Gradient Boosting model using RandomizedSearchCV.
Optimized parameters:
n_estimators: 100
learning_rate: 0.1
max_depth: 5
subsample: 0.8
The tuned model maintained an RMSE of 0.90.
Results
Best Model: Gradient Boosting Regressor (tuned).
Performance: RMSE = 0.90 (well below the target of 3).
Feature Importance:
The most significant features were max_temp and min_temp, as expected.
(Include the image if possible)

Tools and Libraries
Python: Data processing and model development.
scikit-learn: Regression models, hyperparameter tuning.
MLflow: Experiment tracking and model management.
pandas, numpy: Data manipulation.
matplotlib, seaborn: Visualization.
Conclusion
This project demonstrates a comprehensive pipeline for predicting mean temperature using machine learning. The integration of advanced models, hyperparameter tuning, and experiment tracking with MLflow highlights a robust approach to weather prediction. The final tuned Gradient Boosting model achieves excellent accuracy, providing a valuable tool for temperature forecasting in London.

# london_weather_prediction
