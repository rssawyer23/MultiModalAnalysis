import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def parse_df(filename, omit_list):
    df = pd.read_csv(filename)
    if "TestSubject" in df.columns:
        df["Condition"] = df["TestSubject"].apply(lambda cell: float(cell[5]))

        full_rows = df.loc[:, "Condition"] == 1.0
        df = df.loc[full_rows, :]

        remove_rows = df["TestSubject"].apply(lambda cell: cell not in omit_list)
        df = df.loc[remove_rows, :]
        df.index = list(range(df.shape[0]))
    return df


def cross_val_r2_multiple(response, ind_df):
    lm = LinearRegression()
    rlm = Ridge()
    lm_errors = []
    rlm_errors = []
    cv_errors = []
    for i in range(len(response)):
        scaler = StandardScaler()
        indices = list(range(len(response)))
        train_indices = indices[:i] + indices[(i+1):]
        train_y = np.array(response.loc[train_indices]).reshape(-1, 1)
        train_x = np.array(scaler.fit_transform(ind_df.loc[train_indices, :]))
        lm.fit(X=train_x, y=train_y)
        rlm.fit(X=train_x, y=train_y)

        lm_errors.append((response[i] - lm.predict(scaler.transform(ind_df.loc[i, :].values.reshape(1, -1)))))
        rlm_errors.append((response[i] - rlm.predict(scaler.transform(ind_df.loc[i, :].values.reshape(1, -1)))))
        cv_errors.append(response[i] - train_y.mean())
    lm_rss = np.sum(np.array(lm_errors) ** 2)
    rlm_rss = np.sum(np.array(rlm_errors) ** 2)
    tss = np.sum((response - response.mean()) ** 2)
    cv_tss = np.sum(np.array(cv_errors) ** 2)
    lm_r2 = 1.0 - lm_rss / tss
    rlm_r2 = 1.0 - rlm_rss / tss
    cv_lm_r2 = 1.0 - lm_rss / cv_tss
    cv_rlm_r2 = 1.0 - rlm_rss / cv_tss
    print("Multiple Linear Model R2: %.4f CVR2: %.4f" % (lm_r2, cv_lm_r2))
    print("Ridge Regression Model R2: %.4f CVR2: %.4f" % (rlm_r2, cv_rlm_r2))


def cross_val_r2(response, ind_var):
    lm = LinearRegression()
    rlm = Ridge()
    lm_errors = []
    rlm_errors = []
    cv_errors = []
    for i in range(len(response)):
        train_y = np.array(list(response)[:i] + list(response)[(i+1):]).reshape(-1, 1)
        train_x = np.array(list(ind_var)[:i] + list(ind_var)[(i+1):]).reshape(-1, 1)
        lm.fit(X=train_x, y=train_y)
        rlm.fit(X=train_x, y=train_y)

        lm_errors.append((response[i] - lm.predict(ind_var[i]))[0])
        rlm_errors.append((response[i] - rlm.predict(ind_var[i]))[0])
        cv_errors.append((response[i] - train_y.mean()))
    lm_rss = np.sum(np.array(lm_errors)**2)
    rlm_rss = np.sum(np.array(rlm_errors)**2)
    tss = np.sum((response - response.mean())**2)
    cv_tss = np.sum(np.array(cv_errors)**2)
    lm_r2 = 1.0 - lm_rss / tss
    cv_lm_r2 = 1.0 - lm_rss / cv_tss
    rlm_r2 = 1.0 - rlm_rss / tss
    cv_rlm_r2 = 1.0 - rlm_rss / cv_tss
    r, p = pearsonr(response, ind_var)
    print("r:%.4f (%.4f) Linear Model R2:%.4f, CVR2:%.4f" % (r, p, lm_r2, cv_lm_r2))
    #print("Ridge Regres R2:%.4f, CVR2:%.4f" % (rlm_r2, cv_rlm_r2))


def merge_metrics(act_sum_fn, metrics_fn, out_fn="C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output318/ActivitySummary/ActivitySummaryCumulativesRevisedPhases.csv"):
    mdf = pd.read_csv(metrics_fn)
    adf = pd.read_csv(act_sum_fn)
    act_df = pd.read_csv("C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output318/ActivitySummary/ActivitySummaryAppendedRevisedEdited.csv")
    act_df.index = act_df["TestSubject"]
    df = adf.join(mdf)
    df["BaselineDist"] = df["PC1"] - df.loc["GOLD", "PC1"]
    df = df.join(act_df.loc[:, ["RevisedNLG", "FinalGameScore", "Duration"]])
    df = df.drop("GOLD")
    df.to_csv(out_fn)


activity_summary_filename = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output318/ActivitySummary/ActivitySummaryAppendedRevisedEdited.csv"
alt_act_sum_filename = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output318/ActivitySummary/ActivitySummaryCumulativesRevisedPhases.csv"

omit_list = ["CI1301PN042", "CI1301PN043", "CI1301PN035", "CI1301PN037", "CI1301PN073"]
act_df = parse_df(activity_summary_filename, omit_list=omit_list)
alt_df = parse_df(alt_act_sum_filename, omit_list=omit_list)

# Summary statistics for the Study Description paragraph
print(act_df.loc[:, ["Age", "Gender", "RevisedPre", "RevisedPost", "RevisedNLG", "FinalGameScore", "Duration"]].describe())

# Summary statistics of the cumulative actions and durations
# print((alt_df["RevisedDuration"]/60.0).describe())
print(alt_df.loc[:, ["C-A-Conversation", "C-A-BooksAndArticles", "C-A-Worksheet", "C-A-PlotPoint", "C-A-WorksheetSubmit", "C-A-Scanner"]].describe())

# merge_metrics prior to calling below code
phase_df_fn = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output318/ActivitySummary/ActivitySummaryCumulativesRevisedPhases.csv"
phase_df = pd.read_csv(phase_df_fn)

for key in ["Slope-All", "Slope-Tutorial", "Slope-PreScan", "Slope-PostScan", "Dist-All", "Dist-Tutorial", "Dist-PreScan", "Dist-PostScan"]:
    print(key + "-NLG")
    cross_val_r2(phase_df["RevisedNLG"], phase_df[key])
    print("------------------------------------------------------------------")

for key in ["Slope-All", "Slope-Tutorial", "Slope-PreScan", "Slope-PostScan", "Dist-All", "Dist-Tutorial", "Dist-PreScan", "Dist-PostScan"]:
    print(key + "-FinalGameScore")
    cross_val_r2(phase_df["FinalGameScore"], phase_df[key])
    print("------------------------------------------------------------------")
cross_val_r2(phase_df["RevisedNLG"], phase_df["BaselineDist"])
cross_val_r2(phase_df["RevisedNLG"], phase_df["FinalGameScore"])
cross_val_r2(phase_df["BaselineDist"], phase_df["FinalGameScore"])

# Phase slope descriptive statistics
print(phase_df.loc[:, ["Slope-All", "Slope-Tutorial", "Slope-PreScan", "Slope-PostScan"]].describe())
pearsonr(phase_df.loc[:, "Slope-All"], phase_df.loc[:, "RevisedNLG"])
pearsonr(phase_df.loc[:, "Slope-Tutorial"], phase_df.loc[:, "RevisedNLG"])
pearsonr(phase_df.loc[:, "Slope-PreScan"], phase_df.loc[:, "RevisedNLG"])
pearsonr(phase_df.loc[:, "Slope-PostScan"], phase_df.loc[:, "RevisedNLG"])

ttest_ind(phase_df.loc[:, "Slope-PreScan"], phase_df.loc[:, "Slope-PostScan"])
s1 = phase_df.loc[:, "Slope-PreScan"].std()
s2 = phase_df.loc[:, "Slope-PostScan"].std()
n0 = phase_df.shape[0] - 1
s = np.sqrt((n0 * s1**2 + n0 * s2 ** 2)/(2*n0))
d = (phase_df.loc[:, "Slope-PreScan"].mean() - phase_df.loc[:, "Slope-PostScan"].mean()) / s
pearsonr(phase_df.loc[:, "Slope-PreScan"], phase_df.loc[:, "Slope-PostScan"])

# Phase distance descriptive statistics
print(phase_df.loc[:, ["Dist-All", "Dist-Tutorial", "Dist-PreScan", "Dist-PostScan"]].describe())
pearsonr(phase_df.loc[:, "Dist-All"], phase_df.loc[:, "RevisedNLG"])
pearsonr(phase_df.loc[:, "Dist-Tutorial"], phase_df.loc[:, "RevisedNLG"])
pearsonr(phase_df.loc[:, "Dist-PreScan"], phase_df.loc[:, "RevisedNLG"])
pearsonr(phase_df.loc[:, "Dist-PostScan"], phase_df.loc[:, "RevisedNLG"])

pearsonr(phase_df.loc[:, "BaselineDist"], phase_df.loc[:, "RevisedNLG"])

pearsonr(phase_df.loc[:, "RevisedNLG"], phase_df.loc[:, "FinalGameScore"])

pearsonr(phase_df.loc[:, "Slope-All"], phase_df.loc[:, "FinalGameScore"])
pearsonr(phase_df.loc[:, "Dist-All"], phase_df.loc[:, "FinalGameScore"])

cross_val_r2_multiple(response=alt_df["RevisedNLG"], ind_df=alt_df.loc[:, ["Duration", "C-A-Conversation", "C-A-BooksAndArticles", "C-A-Worksheet", "C-A-PlotPoint", "C-A-WorksheetSubmit", "C-A-Scanner"]])



