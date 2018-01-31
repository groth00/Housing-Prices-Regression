import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, Imputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from scipy.stats import skew


df_train = pd.read_csv('~/Desktop/kaggle/train.csv')
df_test = pd.read_csv('~/Desktop/kaggle/test.csv')

#boxplot of year built vs sale price
# plt.figure(figsize=(15, 8))
# sns.boxplot(df_train.YearBuilt, df_train.SalePrice)
# plt.show()


#scatterplot of house area vs price
# plt.figure(figsize=(12,6))
# plt.scatter(x=df_train.GrLivArea, y=df_train.SalePrice)
# plt.xlabel("GrLivArea", fontsize=13)
# plt.ylabel("SalePrice", fontsize=13)
# plt.ylim(0,800000)
# plt.show()

#drop two outliers
df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index, inplace=True)

#merge into one dataset, drop column with sample number
df = pd.concat([df_train, df_test], ignore_index=True)
df.drop(['Id'], axis=1, inplace=True)
print(df.shape)


missing_values = df.isnull().sum()
print(missing_values[missing_values>0].sort_values(ascending=False))
# print(df.groupby(['Neighborhood'])[['LotFrontage']].agg(['mean','median','count']))
df['LotAreaCut'] = pd.qcut(df.LotArea, 10)
# print(df.groupby(['LotAreaCut'])[['LotFrontage']].agg(['mean', 'median', 'count']))

#fill in missing values of LotFrontage by using median value of the lot area of several neighborhoods
# print(df.groupby(['LotAreaCut','Neighborhood'])['LotFrontage'].agg(['mean', 'median', 'count']))
df['LotFrontage']=df.groupby(['LotAreaCut','Neighborhood'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))

#fill in missing values without neighborhood
df['LotFrontage']=df.groupby(['LotAreaCut'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))

#fill in missing values based on feature
cols=["MasVnrArea", "BsmtUnfSF", "TotalBsmtSF", "GarageCars", "BsmtFinSF2", "BsmtFinSF1", "GarageArea"]
for col in cols:
    df[col].fillna(0, inplace=True)

cols1 = ["PoolQC" , "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageQual", "GarageCond", "GarageFinish",
         "GarageYrBlt", "GarageType", "BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType2", "BsmtFinType1",
         "MasVnrType"]
for col in cols1:
    df[col].fillna("None", inplace=True)


cols2 = ["MSZoning", "BsmtFullBath", "BsmtHalfBath", "Utilities", "Functional", "Electrical", "KitchenQual",
         "SaleType","Exterior1st", "Exterior2nd"]
for col in cols2:
    df[col].fillna(df[col].mode()[0], inplace=True)


#convert some numerical features into categorical
NumStr = ["MSSubClass","BsmtFullBath","BsmtHalfBath","HalfBath","BedroomAbvGr","KitchenAbvGr","MoSold","YrSold",
          "YearBuilt","YearRemodAdd","LowQualFinSF","GarageYrBlt"]
for col in NumStr:
    df[col]=df[col].astype(str)


#look at relation of saleprice to these variables and try to map them
df.groupby(['MSSubClass'])[['SalePrice']].agg(['mean','median','count'])


#mapping individual features based on mean/median values
def map_values():
    df["oMSSubClass"] = df.MSSubClass.map({'180': 1,
                                            '30': 2, '45': 2,
                                            '190': 3, '50': 3, '90': 3,
                                            '85': 4, '40': 4, '160': 4,
                                            '70': 5, '20': 5, '75': 5, '80': 5, '150': 5,
                                            '120': 6, '60': 6})

    df["oMSZoning"] = df.MSZoning.map({'C (all)': 1, 'RH': 2, 'RM': 2, 'RL': 3, 'FV': 4})

    df["oNeighborhood"] = df.Neighborhood.map({'MeadowV': 1,
                                                'IDOTRR': 2, 'BrDale': 2,
                                                'OldTown': 3, 'Edwards': 3, 'BrkSide': 3,
                                                'Sawyer': 4, 'Blueste': 4, 'SWISU': 4, 'NAmes': 4,
                                                'NPkVill': 5, 'Mitchel': 5,
                                                'SawyerW': 6, 'Gilbert': 6, 'NWAmes': 6,
                                                'Blmngtn': 7, 'CollgCr': 7, 'ClearCr': 7, 'Crawfor': 7,
                                                'Veenker': 8, 'Somerst': 8, 'Timber': 8,
                                                'StoneBr': 9,
                                                'NoRidge': 10, 'NridgHt': 10})

    df["oCondition1"] = df.Condition1.map({'Artery': 1,
                                               'Feedr': 2, 'RRAe': 2,
                                               'Norm': 3, 'RRAn': 3,
                                               'PosN': 4, 'RRNe': 4,
                                               'PosA': 5, 'RRNn': 5})

    df["oBldgType"] = df.BldgType.map({'2fmCon': 1, 'Duplex': 1, 'Twnhs': 1, '1Fam': 2, 'TwnhsE': 2})

    df["oHouseStyle"] = df.HouseStyle.map({'1.5Unf': 1,
                                               '1.5Fin': 2, '2.5Unf': 2, 'SFoyer': 2,
                                               '1Story': 3, 'SLvl': 3,
                                               '2Story': 4, '2.5Fin': 4})

    df["oExterior1st"] = df.Exterior1st.map({'BrkComm': 1,
                                                 'AsphShn': 2, 'CBlock': 2, 'AsbShng': 2,
                                                 'WdShing': 3, 'Wd Sdng': 3, 'MetalSd': 3, 'Stucco': 3, 'HdBoard': 3,
                                                 'BrkFace': 4, 'Plywood': 4,
                                                 'VinylSd': 5,
                                                 'CemntBd': 6,
                                                 'Stone': 7, 'ImStucc': 7})

    df["oMasVnrType"] = df.MasVnrType.map({'BrkCmn': 1, 'None': 1, 'BrkFace': 2, 'Stone': 3})

    df["oExterQual"] = df.ExterQual.map({'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})

    df["oFoundation"] = df.Foundation.map({'Slab': 1,
                                               'BrkTil': 2, 'CBlock': 2, 'Stone': 2,
                                               'Wood': 3, 'PConc': 4})

    df["oBsmtQual"] = df.BsmtQual.map({'Fa': 2, 'None': 1, 'TA': 3, 'Gd': 4, 'Ex': 5})

    df["oBsmtExposure"] = df.BsmtExposure.map({'None': 1, 'No': 2, 'Av': 3, 'Mn': 3, 'Gd': 4})

    df["oHeating"] = df.Heating.map({'Floor': 1, 'Grav': 1, 'Wall': 2, 'OthW': 3, 'GasW': 4, 'GasA': 5})

    df["oHeatingQC"] = df.HeatingQC.map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

    df["oKitchenQual"] = df.KitchenQual.map({'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})

    df["oFunctional"] = df.Functional.map(
        {'Maj2': 1, 'Maj1': 2, 'Min1': 2, 'Min2': 2, 'Mod': 2, 'Sev': 2, 'Typ': 3})

    df["oFireplaceQu"] = df.FireplaceQu.map({'None': 1, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

    df["oGarageType"] = df.GarageType.map({'CarPort': 1, 'None': 1,
                                               'Detchd': 2,
                                               '2Types': 3, 'Basment': 3,
                                               'Attchd': 4, 'BuiltIn': 5})

    df["oGarageFinish"] = df.GarageFinish.map({'None': 1, 'Unf': 2, 'RFn': 3, 'Fin': 4})

    df["oPavedDrive"] = df.PavedDrive.map({'N': 1, 'P': 2, 'Y': 3})

    df["oSaleType"] = df.SaleType.map({'COD': 1, 'ConLD': 1, 'ConLI': 1, 'ConLw': 1, 'Oth': 1, 'WD': 1,
                                           'CWD': 2, 'Con': 3, 'New': 3})

    df["oSaleCondition"] = df.SaleCondition.map(
        {'AdjLand': 1, 'Abnorml': 2, 'Alloca': 2, 'Family': 2, 'Normal': 3, 'Partial': 4})
map_values()


#drop LotAreaCut (created earlier) and the target feature
df.drop("LotAreaCut",axis=1,inplace=True)
df.drop(['SalePrice'],axis=1,inplace=True)


###PIPELINE

#encode the features with year values
class labelenc(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        lab = LabelEncoder()
        X["YearBuilt"] = lab.fit_transform(X["YearBuilt"])
        X["YearRemodAdd"] = lab.fit_transform(X["YearRemodAdd"])
        X["GarageYrBlt"] = lab.fit_transform(X["GarageYrBlt"])
        return X

#normalize the skewed features and get labels
class skew_dummies(BaseEstimator, TransformerMixin):
    def __init__(self, skew=0.5):
        self.skew = skew

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_numeric = X.select_dtypes(exclude=["object"])
        skewness = X_numeric.apply(lambda x: skew(x))
        skewness_features = skewness[abs(skewness) >= self.skew].index
        X[skewness_features] = np.log1p(X[skewness_features])
        X = pd.get_dummies(X)
        return X


#copy dataset, then fit the copy through the pipeline
pipe = Pipeline([('labenc', labelenc()), ('skew_dummies', skew_dummies(skew=1)),])
df_copy = df.copy()
data_pipe = pipe.fit_transform(df_copy)
print(data_pipe.shape)


#deal with outliers?
scaler = RobustScaler()
n_train=df_train.shape[0]

X = data_pipe[:n_train]
test_X = data_pipe[n_train:]
y= df_train.SalePrice

X_scaled = scaler.fit(X).transform(X)
y_log = np.log(df_train.SalePrice)
test_X_scaled = scaler.transform(test_X)


