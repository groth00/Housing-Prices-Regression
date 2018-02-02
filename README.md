# kaggle

Dataset:
https://www.kaggle.com/c/house-prices-advanced-regression-techniques


PREPROCESSING THE DATASET
-------------------------
Drop two outliers in 'GrLivArea' that have large area, but low house value. (This is inconsistent with the linear trend)

Join the train and test set.

Get the amount of missing values using isnull().sum()
Fill in missing values of LotFrontage (important numerical variable because of its relation of area to sale price) by filling using the median values of the neighborhoods (partition 'LotArea', group LotFrontage by LotArea and Neighborhood)
Fill in missing values of categorical variables with the string 'None' 
Fill in numerical values based on the feature (features with minimal amount of missing values are replaced by the mode, others are replaced with integer 0)

Convert some numerical features into categorical features (for example, numbers used as categories rather than taking a meaningful value)
Map categorical features - for feature 'MSSubClass', the mapping is based on relation of the feature to the SalePrice
note: can map several values of a categorical feature to the same integer

Drop column created earlier to approximate missing values of LotFrontage
Also drop SalePrice since there are only values for the training set


Next time, remove features with significant amount of missing values.
Maybe try OHE or other alternatives instead of manually mapping (however this could be less accurate of course)


PIPELINE
--------
Define label encoder for features that have years as values
Define skew correction method to normalize skewed numerical features (and return mapped values)
Copy dataset and fit through pipeline
Partition dataset based on training and test values, as well as initialize target variable
Standardize data using RobustScaler

Use Lasso or other model to get and visualize feature importance


MODELS
------
Define scoring method
Pick models, name them, get preliminary score
Use grid search to find optimal hyperparameters for each model (adjust alpha, iterations, C, etc..)
More advanced modeling later....



Updated: 2/2/18
