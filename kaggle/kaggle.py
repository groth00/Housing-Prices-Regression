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


#find which columns have a significant amount of missing values
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# print(missing_data.head(n=19))


#drop columns with missing values
#for the column 'Electrical' only drop the 1 missing observation
drop_labels = missing_data[missing_data['Total']>1].index
df_train = df.drop(drop_labels, axis=1)

missing_value = df_train.loc[df_train['Electrical'].isnull()].index
df2 = df_train.drop(missing_value)

#check to make sure the columns were dropped
# print(df_train.head(n=3))
# print(df2.head(n=3))


#encode the categorical variables and join with numerical variables
categorical_variables = df2.select_dtypes(exclude=np.number)
numerical_variables = df2.select_dtypes(include=np.number)
encoded_variables = pd.get_dummies(categorical_variables)

#check to make sure the dimensions will match
print(numerical_variables.shape)
print(encoded_variables.shape)

processed_df = pd.concat([numerical_variables, encoded_variables], axis=1)
print(processed_df.head(n=5))


#split the training data and fit through pipeline
# train_test_split(processed_df)
pipe_dt = make_pipeline(StandardScaler(), PCA(n_components=2), DecisionTreeRegressor(random_state=1))
pipe_dt.fit(processed_df)


#encoding one categorical variable
# print(categorical_variables.iloc[:, 0].head(n=5))
# int_labels = label.fit_transform(categorical_variables.iloc[:, 0]).reshape(-1, 1)
# ohe_labels = ohe.fit_transform(int_labels)
# np.set_printoptions(threshold=None)
# print(ohe_labels.toarray())




#TODO:
#standardize values
#feature selection/PCA
#k-fold cross validation?
#grid search to tune hyperparameters
#pipeline to test models
