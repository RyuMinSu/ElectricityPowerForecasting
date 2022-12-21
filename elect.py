#%%
import numpy as np
import pandas as pd
import datetime

from electPlotMD import *
from electMLMD import *
from electDBMD import *
#%%
host = ""
id = ""
pw = ""
dbName = ""
rsql1 = ""
rsql2 = ""

powercomp = readTb(host, id, pw, dbName, rsql1)
proc = readTb(host, id, pw, dbName, rsql2)


df = powercomp.iloc[:, :-2]
df.columns = df.columns.str.lower() #컬럼 소문자로
df.rename(columns={"zone 1 power consumption": "target"}, inplace=True)
print(f"df shape: {df.shape}")


#####결측처리: 없음(완료)
#####레이블링 및 더미처리
df = sepDatecols(df, "datetime")

#####파생변수생성()
labels = [1, 2, 3, 4, 5]
df = colsQcut(df, "humidity", "humidity_qcut", labels, True)

##변수별 시각화
sns.countplot(data=df, x="day_name")
plt.title("Day Name Count", size=15)

plotList = ["month", "hour", "day_name"]
fig = plt.figure(figsize=(10, 20))
for idx, li in enumerate(plotList):
    plt.subplot(3, 1, idx+1)
    sns.pointplot(data=df, x=plotList[idx], y="target", ci=None)
    plt.title(f"{plotList[idx]} - target Count", size=15)
plt.tight_layout()
plt.legend()
plt.show()

plt.figure(figsize=(20, 10))
sns.pointplot(data=df, x="hour", y="target", hue="day_name", ci=None)
plt.title("hour - taraget: day_name", size=15)
plt.legend(loc="best")
plt.show()

##correlation
mask = np.zeros_like(df.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(10, 10))
sns.heatmap(df.corr(), mask=mask, vmin=-1, vmax=1, annot=True, linewidth=0.5, cmap="coolwarm")
plt.title("Featurers-target Correlation", size=15)

#high correlation boxplot
boxList = ["temperature"]
plt.figure(figsize=(10, 10))
sns.boxplot(df["temperature"])
plt.title("temperature boxplot", size=15)
plt.show()

#####스케일링(day_name, 더미변환)
#컬럼제거: datetime, year, humidity_cut
df = df.drop(["datetime", "year", "humidity_cut"], axis=1)

#더미처리: day_name, humidity_qcut
dumDf = pd.get_dummies(df[["day_name", "humidity_qcut"]])
df = pd.concat([df, dumDf], axis=1)
df = df.drop(["day_name", "humidity_qcut"], axis=1)
print(df.shape)

#스케일링: target빼고 전부
dfCols = list(df)
dfCols.remove("target")

mms = MinMaxScaler()
df[dfCols] = mms.fit_transform(df[dfCols])

X = df.drop(["target"], axis=1)
y = df["target"]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)
print(x_train.shape, y_train.shape)

#####모델학습 및 평가
linReg = LinearRegression()
ranReg = RandomForestRegressor()
votReg = VotingRegressor([
    ("linear", linReg),
    ("randomforest", ranReg)
])
xgbReg = xgboost.XGBRegressor()
lgbReg = lightgbm.LGBMRegressor()

models = [linReg, ranReg, votReg, xgbReg, lgbReg]
MLPerformance(x_train, x_test, y_train, y_test, models)

#####learning curve
trainSizes = np.linspace(.1, 1.0, 5)
cv = 3
scoring = "neg_mean_squared_error"
plotLC(x_train, y_train, trainSizes, cv, scoring, models)

#####feature Importance
plotFI(ranReg, x_train)
