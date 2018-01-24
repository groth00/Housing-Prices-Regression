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
df_train = df.drop(missing_data[missing_data['Total']>1].index, axis=1)
df_train = df.drop(df_train.loc[df_train['Electrical'].isnull()].index)

print(df_train.head(n=3))

#want to encode the categorical variables, so we don't care about the numerical ones for now
categorical_variables = df.select_dtypes(exclude=np.number)



