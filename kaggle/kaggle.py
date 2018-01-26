import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


ohe = OneHotEncoder()
label = LabelEncoder()
sc = StandardScaler()

df = pd.read_csv('~/Desktop/kaggle/train.csv')
df_test = pd.read_csv('~/Desktop/kaggle/test.csv')
# 81 columns, first one is the sample number

# need to isolate the SalePrices (what we are trying to predict)
y = df.iloc[:, -1]


#missing values for training set
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# print(missing_data.head(n=19))


#missing values for test set
total_test = df_test.isnull().sum().sort_values(ascending=False)
percent_test = (df_test.isnull().sum()/df_test.isnull().count()).sort_values(ascending=False)
missing_data_test = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# print(missing_data_test.head(n=19))



#drop columns with missing values and single observation in Electrical column
drop_labels = missing_data[missing_data['Total']>1].index
df_train = df.drop(drop_labels, axis=1)

drop_labels_test = missing_data[missing_data['Total']>1].index
df_test = df_test.drop(drop_labels_test, axis=1)

missing_value = df_train.loc[df_train['Electrical'].isnull()].index
df_train = df_train.drop(missing_value)
y = y.drop(missing_value)
y = y.values

missing_value_test = df_test.loc[df_test['Electrical'].isnull()].index
df_test = df_test.drop(missing_value_test)


#drop the SalesPrice column from training set since we are trying to predict it
df_train = df_train.drop(df_train.columns[-1], axis=1)

#check to make sure the columns were dropped
# print(df_train.head(n=3))
# print(df_test.head(n=3))


#dimensions: 1459 x 62
# print(df_train.shape)

#encode training dataset
categorical_variables = df_train.select_dtypes(exclude=np.number)
numerical_variables = df_train.select_dtypes(include=np.number)
encoded_variables = pd.get_dummies(categorical_variables)

#encode test dataset
categorical_variables_test = df_test.select_dtypes(exclude=np.number)
numerical_variables_test = df_test.select_dtypes(include=np.number)
encoded_variables_test = pd.get_dummies(categorical_variables_test)


#join encoded columns with numerical columns
processed_df = pd.concat([numerical_variables, encoded_variables], axis=1)
processed_df_test = pd.concat([numerical_variables_test, encoded_variables_test], axis=1)

#view data frame entirely
# pd.set_option('display.max_columns', 999)
# print(processed_df.head(n=5))



#Currently using SelectKBest for feature selection, can experiment later
X = processed_df.iloc[:, 1:]
X_new = SelectKBest(f_regression, k=10).fit_transform(X, y)
print(X.shape, X_new.shape, y.shape)


#split the training data and traing with random forest
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.4, random_state=1)
print(X_train, '\n' * 5, X_test, '\n' * 5, y_train, '\n' * 5, y_test)
forest = RandomForestRegressor(n_estimators=1000, criterion='mse', random_state=1, n_jobs=-1)
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred),
                                       mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred),
                                       r2_score(y_test, y_test_pred)))



# Random Forest returns ranking of features, in the future can try to select these manually before training
features = forest.feature_importances_
print(features)
print('\n')





#TODO:
#input test data and preprocess that also
#use to pipline to standardize values -> feature selection/PCA -> train model
#k-fold cross validation?
#grid search to tune hyperparameters
#test how accurate the model is on the test data
