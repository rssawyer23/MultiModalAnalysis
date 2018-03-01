import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr
from scipy.stats import ttest_ind


def parse_df(filename, omit_list):
    df = pd.read_csv(filename)
    df["Condition"] = df["TestSubject"].apply(lambda cell: float(cell[5]))

    full_rows = df.loc[:, "Condition"] == 1.0
    df = df.loc[full_rows, :]

    remove_rows = df["TestSubject"].apply(lambda cell: cell not in omit_list)
    df = df.loc[remove_rows, :]
    df.index = list(range(df.shape[0]))
    return df


activity_summary_filename = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/ActivitySummary/ActivitySummaryGraded.csv"
alt_act_sum_filename = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/ActivitySummary/ActivitySummaryCumulativesRevisedSlope.csv"

omit_list = ["CI1301PN042", "CI1301PN043"]
act_df = parse_df(activity_summary_filename, omit_list=omit_list)
alt_df = parse_df(alt_act_sum_filename, omit_list=omit_list)

# Summary statistics for the Study Description paragraph
act_df["Age"].describe()
act_df["Gender"].describe()
act_df["PreTestScore"].describe()
act_df["PostTestScore"].describe()

# Summary statistics of the cumulative actions and durations
(alt_df["RevisedDuration"]/60.0).describe()
alt_df["C.A.Conversation"].describe()
alt_df["C.A.BooksAndArticles"].describe()
alt_df["C.A.Worksheet"].describe()
alt_df["C.A.PlotPoint"].describe()
alt_df["C.A.WorksheetSubmit"].describe()
alt_df["C.A.Scanner"].describe()

phase_df_fn = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/ActivitySummary/ActivitySummaryCumulativesRevisedPhases.csv"
phase_df = pd.read_csv(phase_df_fn)
full_rows = phase_df.loc[:,"Condition"] == 1.0
phase_df = phase_df.loc[full_rows, :]

# Phase slope descriptive statistics
phase_df.loc[:,["Slope-All","Slope-Tutorial", "Slope-PreScan", "Slope-PostScan"]].describe()
pearsonr(phase_df.loc[:,"Slope-All"], phase_df.loc[:, "NLG"])
pearsonr(phase_df.loc[:,"Slope-Tutorial"], phase_df.loc[:, "NLG"])
pearsonr(phase_df.loc[:,"Slope-PreScan"], phase_df.loc[:, "NLG"])
pearsonr(phase_df.loc[:,"Slope-PostScan"], phase_df.loc[:, "NLG"])

ttest_ind(phase_df.loc[:, "Slope-PreScan"], phase_df.loc[:, "Slope-PostScan"])
s1 = phase_df.loc[:, "Slope-PreScan"].std()
s2 = phase_df.loc[:, "Slope-PostScan"].std()
n0 = phase_df.shape[0] - 1
s = np.sqrt((n0 * s1**2 + n0 * s2 ** 2)/(2*n0))
d = (phase_df.loc[:, "Slope-PreScan"].mean() - phase_df.loc[:, "Slope-PostScan"].mean()) / s
pearsonr(phase_df.loc[:, "Slope-PreScan"], phase_df.loc[:, "Slope-PostScan"])

# Phase distance descriptive statistics
phase_df.loc[:, ["Dist-All", "Dist-Tutorial", "Dist-PreScan", "Dist-PostScan"]].describe()
pearsonr(phase_df.loc[:,"Dist-All"], phase_df.loc[:, "NLG"])
pearsonr(phase_df.loc[:,"Dist-Tutorial"], phase_df.loc[:, "NLG"])
pearsonr(phase_df.loc[:,"Dist-PreScan"], phase_df.loc[:, "NLG"])
pearsonr(phase_df.loc[:,"Dist-PostScan"], phase_df.loc[:, "NLG"])

pearsonr(phase_df.loc[:,"FinalDist"], phase_df.loc[:, "NLG"])

pearsonr(phase_df.loc[:,"NLG"], phase_df.loc[:, "FinalGameScore"])

pearsonr(phase_df.loc[:, "Slope-All"], phase_df.loc[:, "FinalGameScore"])
pearsonr(phase_df.loc[:, "Dist-All"], phase_df.loc[:, "FinalGameScore"])
