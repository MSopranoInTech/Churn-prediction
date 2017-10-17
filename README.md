# Churn Prediction - Ride Sharing Services 

Predicted possibility of customer churning, thereby targeting special campaigns to increase retention rate. Built random forest model to predict churn and developed profit curves to tune parameters. Improved recall rate by 15% from the baseline model through feature reduction, decreasing false predictions.

## Motivation
Customer churning is one of the biggest problems faced in the industry, and trying to predict churn rate is one of the common applications of data science in the industry. Predicting churn and possible characteristics of the churning customers allows the business to use promotions and offers to lure such customers back in. Replicating such real world scenarios helped me understand how machine learning algorithms are implemented and the difficulties they face. 

## Feature Engineering and EDA
A customer was considered to be active, if he had taken a ride in the last 30 days. While understanding the data, we realized that some features leaked data. Engineering features from them proved to be difficult yet, improved the performance of the model. Also upon EDA, we found that the customers were almost evenly split, hence we didn't have to focus on imbalanced classes and other issues.

## Model Development
We started of with a logistic regression model as a baseline model. Following the CRISP-DM methodology, we iterated over adding multiple features and compared with a Random Forest model. To identify the best model, we performed a grid search and performed a bootstrap operation as well, to see if that had an influence. 

## Result and Inference
The Random forest model performed best, predicting with a recall of 75%, 15% more than th baseline model. 

## Files

* model.py - Compares the performance of models and stores the best model in pickle format

## Credits
This project would not be possible without the efforts of my fellow teammates Joseph Fang, Praveen Raman

