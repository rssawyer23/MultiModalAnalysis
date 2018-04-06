# Script used for testing predictions from EDM2018 Trajectory paper
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

filepath = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/ActivitySummary/ActivitySummaryCumulativesRevisedPhases.csv"
data = pd.read_csv(filepath)
#data = data.iloc[:-1, :]  # Removing Rowe

full_agency_rows = data.loc[:,"Condition"] == 1
data = data.loc[full_agency_rows,:]
data.index = list(range(data.shape[0]))

data["PC1 o D"] = data.loc[:, "PC1"] / data.loc[:, "RevisedDuration"]

action_features = ["C.A.Conversation", "C.A.BooksAndArticles",
                   "C.A.Worksheet", "C.A.PlotPoint", "C.A.WorksheetSubmit",
                   "C.A.Scanner", "RevisedDuration"]
base_features = ["Slope-All"]

full_features = base_features
scaler = StandardScaler()
data.loc[:,full_features] = scaler.fit_transform(data.loc[:,full_features])
response_variable = "NLG"

kfold = KFold(n_splits=data.shape[0])
all_errors = []
cv_mean_errors = []
for train, test in kfold.split(data):
    lm = LinearRegression()
    lm.fit(X=data.loc[train, full_features], y=data.loc[train, response_variable])
    prediction = lm.predict(X=data.loc[test, full_features])
    error = (prediction - data.loc[test, response_variable]).iloc[0]
    all_errors.append(error)

    cv_mean = data.loc[train, response_variable].mean()
    cv_error = (cv_mean - data.loc[test, response_variable]).iloc[0]
    cv_mean_errors.append(cv_error)

print("USING: %s to PREDICT: %s" % (full_features, response_variable))
print(pd.Series(lm.coef_, index=full_features))
mean_errors = data.loc[:,response_variable] - data.loc[:, response_variable].mean()
print("Predicting %d in LOO" % len(mean_errors))
print("MAE: %.4f\tmMAE: %.4f\tcvMAE: %.4f" % (np.mean(np.abs(all_errors)), np.mean(np.abs(mean_errors)), np.mean(np.abs(cv_mean_errors))))
print("MSE: %.4f\tmMSE: %.4f\tcvMSE: %.4f" % (np.mean(np.array(all_errors)**2), np.mean(np.array(mean_errors)**2), np.mean(np.array(cv_mean_errors)**2)))
rss = np.sum(np.array(all_errors)**2)
tss = np.sum(np.array(mean_errors)**2)
cvtss = np.sum(np.array(cv_mean_errors)**2)
print("R2: %.4f" % (1.0 - rss/tss))
print("CVR2: %.4f" % (1.0 - rss/cvtss))
