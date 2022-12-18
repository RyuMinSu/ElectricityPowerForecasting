#create report
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly import express as px
from sklearn.model_selection import learning_curve
from dataprep.eda import create_report

def createReport(df, reportName):
    report = create_report(df)
    report.save(reportName)

#learning curve
def plotLC(x_train, y_train, trainSizes, cv, scoring, modelList=None):
    fig = plt.figure(figsize=(15, 10))
    for idx, model in enumerate(modelList):
        trainSizes, trainScores, testScores = learning_curve(model, x_train, y_train, train_sizes=trainSizes, cv=cv, scoring=scoring, n_jobs=1)
        trainScoresMean = np.mean(np.sqrt(-trainScores), axis=1)
        testScoresMean = np.mean(np.sqrt(-testScores), axis=1)
        plt.subplot(2, 2, idx+1)
        plt.plot(trainSizes, trainScoresMean, "-o", label="train rmse")
        plt.plot(trainSizes, testScoresMean, "-o", label="test rmse")
        plt.legend()
        plt.title(f"{model.__class__.__name__}", size=15)
    plt.tight_layout()
    plt.xlabel("train size", fontsize=10)
    plt.ylabel("rmse", fontsize=10)
    plt.show()

#plot feature importance
def plotFI(model, x_train):
    ftImp = model.feature_importances_
    ftImpS = pd.Series(ftImp, index=x_train.columns)
    ftImpS = ftImpS.sort_values(ascending=False)
    plt.figure(figsize=(10, 10))
    sns.barplot(x=ftImpS.values[:10], y=ftImpS.index[:10])
    plt.title(f"{model.__class__.__name__} FeatureImportance", size=15)
    plt.xlabel("Importance", fontsize=10)
    plt.ylabel("Features", fontsize=10)
