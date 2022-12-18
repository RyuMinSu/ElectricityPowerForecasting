import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost
import lightgbm

#sepearte date column
def sepDatecols(df, datecol):
    df[datecol] = pd.to_datetime(df[datecol])
    df["year"] = df[datecol].dt.year
    df["month"] = df[datecol].dt.month
    df["day"] = df[datecol].dt.day
    df["hour"] = df[datecol].dt.hour
    df["day_name"] = df[datecol].dt.day_name()
    print(f"seperated {datecol} column df shape: {df.shape}")
    return df

#sector qcut or cut
def colsQcut(df, col, newcol1, labelList=None, cutcol=False):
    if cutcol == True:
        newcol2 = input("cut을 위한 컬럼명을 입력하세요:")
        df[newcol1] = pd.qcut(df[col], 5, labels=labelList)
        df[newcol2] = pd.cut(df[col], 5, labels=labelList)
        print("append qcut or cut columns df shape: {df.shape}")
        print("qcut original categories:")
        print(pd.qcut(df[col].values.tolist(), 5).categories)
        print(df[newcol1].cat.categories[0])
        print("\ncut categories:")
        print(pd.cut(df[col].values.tolist(), 5).categories)
        print(df[newcol2].cat.categories[0])
        return df
    else:
        df[newcol1] = pd.qcut(df[col], 5, labels=labelList)
        print("append qcut or cut columns df shape: {df.shape}")
        print("qcut original categories:")
        print(pd.qcut(df[col].values.tolist(), 5).categories)
        print(df[newcol1].cat.categories[0])
        # print("\ncut categories:")
        # print(pd.cut(df[col].values.tolist(), 5).categories)
        # print(df[newcol2].cat.categories[0])
        return df

#machine learning
def MLPerformance(x_train, x_test, y_train, y_test, modelList=None):
    for model in modelList:
        model.fit(x_train, y_train)
        trainPred = model.predict(x_train)
        testPred = model.predict(x_test)

        trainRmse = mean_squared_error(y_train, trainPred, squared=False)
        trainR2 = r2_score(y_train, trainPred)

        testRmse = mean_squared_error(y_test, testPred, squared=False)
        testR2 = r2_score(y_test, testPred)

        print(f"\n{model.__class__.__name__} train rmse: {trainRmse:.4f}")
        print(f"{model.__class__.__name__} train r2: {trainR2:.4f}")
        print(f"{model.__class__.__name__} test rmse: {testRmse:.4f}")
        print(f"{model.__class__.__name__} test r2: {testR2:.4f}")