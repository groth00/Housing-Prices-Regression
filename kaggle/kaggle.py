import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

ohe = OneHotEncoder()
label = LabelEncoder()

df = pd.read_csv('~/Desktop/kaggle/train.csv')
# 81 columns, first one is the sample number


#find which columns have a significant amount of missing values
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(n=19))


#drop columns with missing values
#for the column 'Electrical' only drop the 1 missing observation
drop_labels = missing_data[missing_data['Total']>1].index
df_train = df.drop(drop_labels, axis=1)

missing_value = df_train.loc[df_train['Electrical'].isnull()].index
df2 = df_train.drop(missing_value)

#check to make sure the columns were dropped
print(df_train.head(n=3))
print(df2.head(n=3))


#want to encode the categorical variables, so we don't care about the numerical ones for now
categorical_variables = df2.select_dtypes(exclude=np.number)


#encoding one categorical variable
# print(categorical_variables.iloc[:, 0].head(n=5))
# int_labels = label.fit_transform(categorical_variables.iloc[:, 0]).reshape(-1, 1)
# ohe_labels = ohe.fit_transform(int_labels)
# np.set_printoptions(threshold=None)
# print(ohe_labels.toarray())


#encode all of the categorical variables
for i in range(categorical_variables.shape[1]):
    int_labels = label.fit_transform(categorical_variables.iloc[:, i]).reshape(-1, 1)
    ohe_labels = ohe.fit_transform(int_labels)

#need to save encoded categorical variables and join with numerical variables



#standardize values
#feature selection/PCA
#maybe k-cross-fold validation?
#grid search to tune hyperparameters
#pipeline to test models
