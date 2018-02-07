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
# print(df.shape)


missing_values = df.isnull().sum()
# print(missing_values[missing_values>0].sort_values(ascending=False))
# print(df.groupby(['Neighborhood'])[['LotFrontage']].agg(['mean','median','count']))
df['LotAreaCut'] = pd.qcut(df.LotArea, 10)
# print(df.groupby(['LotAreaCut'])[['LotFrontage']].agg(['mean', 'median', 'count']))

#fill in missing values of LotFrontage by using median value of the lot area of several neighborhoods
# print(df.groupby(['LotAreaCut','Neighborhood'])['LotFrontage'].agg(['mean', 'median', 'count']))
df['LotFrontage']=df.groupby(['LotAreaCut','Neighborhood'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))

#fill in missing values without neighborhood
df['LotFrontage']=df.groupby(['LotAreaCut'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))

#our own missing values
categorical = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageQual', 'GarageCond', 'GarageFinish',
              'GarageYrBlt', 'GarageType', 'BsmtExposure', 'BsmtCond', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1',
              'MasVnrType']
for col in categorical:
    df[col].fillna('None', inplace=True)

numerical_0 = ['MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageCars', 'BsmtFinSF2', 'BsmtFinSF1', 'GarageArea']
for col in numerical_0:
    df[col].fillna(0, inplace=True)

numerical_mode = ['MSZoning', 'BsmtFullBath', 'BsmtHalfBath', 'Utilities', 'Functional', 'Electrical', 'KitchenQual',
         'SaleType','Exterior1st', 'Exterior2nd']
for col in numerical_mode:
    df[col].fillna(df[col].mode()[0], inplace=True)


# print(df['MSZoning'].mode()[0])

###Feature engineering

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

# print(df.shape)

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
#so when the data is sent through the skew_dummies part of the pipeline, it will only operate on columns of
#the dataset that are not type object, meaning it excludes columns with strings
#therefore, leaving 'None' in the columns before sending it through the pipeline is equivalent to throwing it away

pipe = Pipeline([('labenc', labelenc()), ('skew_dummies', skew_dummies(skew=1)),])
df_copy = df.copy()
# print(df_copy.head(n=5).describe)
data_pipe = pipe.fit_transform(df_copy)
# print(data_pipe.shape)



#deal with outliers
scaler = RobustScaler(copy=False)

#split X into training and test samples, as well as setting y to house prices of training set
n_train=df_train.shape[0]
X = data_pipe[:n_train]
test_X = data_pipe[n_train:]
y= df_train.SalePrice

#standardize values with RobustScaler
X_scaled = scaler.fit(X).transform(X)
y_log = np.log(df_train.SalePrice)
test_X_scaled = scaler.transform(test_X)


#visualize the importance of features
# lasso = Lasso(alpha=0.001)
# lasso.fit(X_scaled, y_log)
# Fl_lasso = pd.DataFrame({'Feature Importance':lasso.coef_}, index=data_pipe.columns)
# Fl_lasso.sort_values("Feature Importance",ascending=False)
# Fl_lasso[Fl_lasso["Feature Importance"]!=0].sort_values("Feature Importance").plot(kind="barh",figsize=(15,25))
# plt.xticks(rotation=90)
# plt.show()


#cross validation strategy
def rmse_cv(model, X, y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5))
    return rmse


#pick models, set names, print scores of each model
models = [LinearRegression(), Ridge(), Lasso(alpha=0.01, max_iter=10000), RandomForestRegressor(),
          GradientBoostingRegressor(), SVR(), LinearSVR(), ElasticNet(alpha=0.001, max_iter=10000),
          SGDRegressor(max_iter=1000, tol=1e-3), BayesianRidge(), KernelRidge(alpha=0.6, kernel='polynomial',
          degree=2, coef0=2.5), ExtraTreesRegressor()]
names = ['LR', 'Ridge', 'Lasso', 'RF', 'GBR', 'SVR', 'LinSVR', 'Ela', 'SGD', 'Bay', 'Ker', 'Extra']
# for name, model in zip(names, models):
#     score = rmse_cv(model, X_scaled, y_log)
#     print('{}: {:.6f}, {:.4f}'.format(name, score.mean(), score.std()))


#define grid search method
# class grid():
#     def __init__(self, model):
#         self.model = model
#
#     def grid_get(self, X, y, param_grid):
#         grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='neg_mean_squared_error',
#                                    return_train_score=True)
#         grid_search.fit(X, y)
#         print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))
#         grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])
#         print(pd.DataFrame(grid_search.cv_results_)[['params', 'mean_test_score', 'std_test_score']])

# grid(Lasso()).grid_get(X_scaled,y_log,{'alpha': [0.0004,0.0005,0.0007,0.0006,0.0009,0.0008],'max_iter':[10000]})
# grid(Ridge()).grid_get(X_scaled,y_log,{'alpha':[35,40,45,50,55,60,65,70,80,90]})
# grid(SVR()).grid_get(X_scaled,y_log,{'C':[11,12,13,14,15],'kernel':["rbf"],"gamma":[0.0003,0.0004],
#                                      "epsilon":[0.008,0.009]})
#
# param_grid={'alpha':[0.2,0.3,0.4,0.5], 'kernel':["polynomial"], 'degree':[3],'coef0':[0.8,1,1.2]}
# grid(KernelRidge()).grid_get(X_scaled,y_log,param_grid)
#
# grid(ElasticNet()).grid_get(X_scaled,y_log,{'alpha':[0.0005,0.0008,0.004,0.005],'l1_ratio':[0.08,0.1,0.3,0.5,0.7],
#                                             'max_iter':[10000]})

#grid search for ensemble models
# class AverageWeight(BaseEstimator, RegressorMixin):
#     def __init__(self, mod, weight):
#         self.mod = mod
#         self.weight = weight
#
#     def fit(self, X, y):
#         self.models_ = [clone(x) for x in self.mod]
#         for model in self.models_:
#             model.fit(X, y)
#         return self
#
#     def predict(self, X):
#         w = list()
#         pred = np.array([model.predict(X) for model in self.models_])
#         # for every data point, single model prediction times weight, then add them together
#         for data in range(pred.shape[1]):
#             single = [pred[model, data] * weight for model, weight in zip(range(pred.shape[0]), self.weight)]
#             w.append(np.sum(single))
#         return w


#choose these models because they performed well
lasso = Lasso(alpha=0.0005,max_iter=10000)
ridge = Ridge(alpha=60)
svr = SVR(gamma= 0.0004,kernel='rbf',C=13,epsilon=0.009)
ker = KernelRidge(alpha=0.2 ,kernel='polynomial',degree=3 , coef0=0.8)
ela = ElasticNet(alpha=0.005,l1_ratio=0.08,max_iter=10000)
bay = BayesianRidge()

#weights of selected models based on gridsearch score
w1 = 0.02
w2 = 0.2
w3 = 0.25
w4 = 0.3
w5 = 0.03
w6 = 0.2

#average score across 6 models
# weight_avg = AverageWeight(mod = [lasso,ridge,svr,ker,ela,bay],weight=[w1,w2,w3,w4,w5,w6])
# rmse_cv(weight_avg,X_scaled,y_log),  rmse_cv(weight_avg,X_scaled,y_log).mean()


#average score across 2 best models
# weight_avg = AverageWeight(mod = [svr,ker],weight=[0.5,0.5])
# rmse_cv(weight_avg,X_scaled,y_log),  rmse_cv(weight_avg,X_scaled,y_log).mean()


class stacking(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, mod, meta_model):
        self.mod = mod
        self.meta_model = meta_model
        self.kf = KFold(n_splits=5, random_state=42, shuffle=True)

    def fit(self, X, y):
        self.saved_model = [list() for i in self.mod]
        oof_train = np.zeros((X.shape[0], len(self.mod)))

        for i, model in enumerate(self.mod):
            for train_index, val_index in self.kf.split(X, y):
                renew_model = clone(model)
                renew_model.fit(X[train_index], y[train_index])
                self.saved_model[i].append(renew_model)
                oof_train[val_index, i] = renew_model.predict(X[val_index])

        self.meta_model.fit(oof_train, y)
        return self

    def predict(self, X):
        whole_test = np.column_stack([np.column_stack(model.predict(X) for model in single_model).mean(axis=1)
                                      for single_model in self.saved_model])
        return self.meta_model.predict(whole_test)

    def get_oof(self, X, y, test_X):
        oof = np.zeros((X.shape[0], len(self.mod)))
        test_single = np.zeros((test_X.shape[0], 5))
        test_mean = np.zeros((test_X.shape[0], len(self.mod)))
        for i, model in enumerate(self.mod):
            for j, (train_index, val_index) in enumerate(self.kf.split(X, y)):
                clone_model = clone(model)
                clone_model.fit(X[train_index], y[train_index])
                oof[val_index, i] = clone_model.predict(X[val_index])
                test_single[:, j] = clone_model.predict(test_X)
            test_mean[:, i] = test_single.mean(axis=1)
        return oof, test_mean

a = Imputer().fit_transform(X_scaled)
b = Imputer().fit_transform(y_log.values.reshape(-1,1)).ravel()

# stack_model = stacking(mod=[lasso,ridge,svr,ker,ela,bay],meta_model=ker)
#
# X_train_stack, X_test_stack = stack_model.get_oof(a,b,test_X_scaled)
# X_train_add = np.hstack((a,X_train_stack))
# X_test_add = np.hstack((test_X_scaled,X_test_stack))

stack_model = stacking(mod=[lasso,ridge,svr,ker,ela,bay],meta_model=ker)
stack_model.fit(a,b)
pred = np.exp(stack_model.predict(test_X_scaled))

#submission

results = pd.DataFrame({'Id':df_test.Id, 'SalePrice':pred})
results.to_csv('submission.csv', index=False)