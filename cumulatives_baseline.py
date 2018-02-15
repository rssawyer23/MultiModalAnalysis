import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

filepath = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/ActivitySummary/ActivitySummaryCumulatives.csv"
data = pd.read_csv(filepath)

action_features = ["C-A-Conversation", "C-A-BooksAndArticles",
                   "C-A-Worksheet", "C-A-PlotPoint", "C-A-WorksheetSubmit",
                   "C-A-Scanner"]
location_features = ["C-L-Dock", "C-L-Beach", "C-L-Outside",
                     "C-L-Infirmary", "C-L-Lab", "C-L-Dining", "C-L-Dorm", "C-L-Bryce"]
full_features = action_features + location_features
scaler = StandardScaler()
data.loc[:,full_features] = scaler.fit_transform(data.loc[:,full_features])
response_variable = "NLG"

kfold = KFold(n_splits=data.shape[0])
all_errors = []
cv_mean_errors = []
for train, test in kfold.split(data):
    lm = Ridge()
    lm.fit(X=data.loc[train, action_features], y=data.loc[train, response_variable])
    prediction = lm.predict(X=data.loc[test, action_features])
    error = (prediction - data.loc[test, response_variable]).iloc[0]
    all_errors.append(error)

    cv_mean = data.loc[train, response_variable].mean()
    cv_error = (cv_mean - data.loc[test, response_variable]).iloc[0]
    cv_mean_errors.append(cv_error)
print(pd.Series(lm.coef_, index=action_features))
mean_errors = data.loc[:,response_variable] - data.loc[:, response_variable].mean()
print("MAE: %.4f\tmMAE: %.4f\tcvMAE: %.4f" % (np.mean(np.abs(all_errors)), np.mean(np.abs(mean_errors)), np.mean(np.abs(cv_mean_errors))))
print("MSE: %.4f\tmMSE: %.4f\tcvMSE: %.4f" % (np.mean(np.array(all_errors)**2), np.mean(np.array(mean_errors)**2), np.mean(np.array(cv_mean_errors)**2)))
rss = np.sum(np.array(all_errors)**2)
tss = np.sum(np.array(mean_errors)**2)
cvtss = np.sum(np.array(cv_mean_errors)**2)
print("R2: %.4f" % (1.0 - rss/tss))
print("CVR2: %.4f" % (1.0 - rss/cvtss))
