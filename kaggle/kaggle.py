import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor


ohe = OneHotEncoder()
label = LabelEncoder()
sc = StandardScaler()

df = pd.read_csv('~/Desktop/kaggle/train.csv')
# 81 columns, first one is the sample number

# need to isolate the SalePrices (what we are trying to predict)
y = (df.iloc[:, -1])



#find which columns have a significant amount of missing values
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# print(missing_data.head(n=19))


#drop columns with missing values
#from the column 'Electrical' drop the 1 observation w/missing value
#also drop the observation from y (SalePrices) to match dimensions
drop_labels = missing_data[missing_data['Total']>1].index
df_train = df.drop(drop_labels, axis=1)

missing_value = df_train.loc[df_train['Electrical'].isnull()].index
df_train = df_train.drop(missing_value)
y = y.drop(missing_value)

df_train = df_train.drop(df_train.columns[-1], axis=1)
#check to make sure the columns were dropped
# print(df_train.head(n=3))

#dimensions: 1459 x 62
# print(df_train.shape)

#encode the categorical variables and join with numerical variables
categorical_variables = df_train.select_dtypes(exclude=np.number)
numerical_variables = df_train.select_dtypes(include=np.number)
encoded_variables = pd.get_dummies(categorical_variables)


#check to make sure the dimensions will match
# print(numerical_variables.shape)
# print(encoded_variables.shape)

#processed data frame is now 1459 x 221
processed_df = pd.concat([numerical_variables, encoded_variables], axis=1)

#view data frame entirely
# pd.set_option('display.max_columns', 999)
# print(processed_df.head(n=5))



#split the training data and fit through pipeline (split data later?)
X = processed_df.values

#currently works, but there is a warning for StandardScaler converting int64 to float64
pipe_dt = make_pipeline(StandardScaler(), PCA(n_components=2), DecisionTreeRegressor(random_state=1))
pipe_dt.fit(X, y)



#TODO:
#input test data and preprocess that also
#use to pipline to standardize values -> feature selection/PCA -> train model
#k-fold cross validation?
#grid search to tune hyperparameters
#test how accurate the model is on the test data
