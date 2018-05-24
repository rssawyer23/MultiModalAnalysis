import pandas as pd
import seaborn as sns
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

ALL_EMOTIONS = ["Negative", "Positive", "Neutral",
                "Anger", "Frustration", "Disgust",
                "Surprise", "Contempt", "Confusion",
                "Joy", "Fear", "Sadness"]


def print_demographics(wdf, adf):
    study_actdf = adf.loc[wdf.index, :]
    students = study_actdf.shape[0]
    females = np.sum(study_actdf["Gender"] == 2)
    print("Total students: %d" % students)
    print("Females: %d (%.1f)" % (students, females/students * 100))
    study_actdf["DurationMin"] = study_actdf["Duration"] / 60
    print(study_actdf.loc[:, ["FinalGameScore", "Age", "DurationMin"]].describe().loc[["mean", "std", "min", "max"]].T.round(1))


def calculate_repeated_measure_differences(wdf, reduced=True, comp_type="DurationProp"):
    modifier_dict = {"After-Scan": ["Positive", "Negative"],
                     "During-Book": ["Relevant", "Irrelevant"],
                     "After-Submission": ["Correct", "Incorrect"]}
    new_cols = []
    for emotion in ALL_EMOTIONS:
        for action in modifier_dict.keys():
            new_cols.append(action+"-"+emotion+"-Difference")
            col_one = wdf.loc[:, action+modifier_dict[action][0]+"-"+emotion+"Evidence-" + comp_type]
            col_two = wdf.loc[:, action+modifier_dict[action][1]+"-"+emotion+"Evidence-" + comp_type]
            wdf.loc[:, action+"-"+emotion+"-Difference"] = col_one - col_two
    if reduced:
        return wdf.loc[:, new_cols]
    else:
        return wdf


def calculate_holm_bon(pvals, alpha=0.05):
    holm_bon = []
    m = len(pvals)
    for i in range(m):
        k = i + 1
        pval = pvals.iloc[i]
        sig_thresh = alpha / (m + 1 - k)
        before_significant = pval <= sig_thresh
        holm_bon.append(before_significant)
    return holm_bon


def calculate_holm_bon_thresholds(pvals, alpha=0.05):
    thresholds = []
    m = len(pvals)
    for i in range(m):
        k = i + 1
        sig_thresh = alpha / (m + 1 - k)
        thresholds.append(sig_thresh)
    return thresholds


def calculate_correlation_table(dif_df, act_df, response):
    # Calculate several pairwise correlations against the response variable contained in the activity summary
    cor_df = pd.DataFrame()
    trows = dif_df.index  # For ensuring the correlations are being calculated on the same rows
    # cols = [e for e in dif_df.columns if "Negative" not in e and "Positive" not in e and "Neutral" not in e]
    cols = [e for e in dif_df.columns if "Frustration" in e or "Confusion" in e or "Joy" in e]
    for col in cols:
        if "Scan" in col:  # Adjusting the rows to only the ones with valid data for action
            srows = ddf["ScanActive"] == 1
        elif "Book" in col:
            srows = ddf["BookActive"] == 1
        elif "Submission" in col:
            srows = ddf["SubmissionActive"] == 1
        else:
            srows = trows
        r, p = ss.pearsonr(dif_df.loc[srows, col], act_df.loc[srows, response])
        n = dif_df.loc[srows, col].shape[0]
        cor_df = pd.concat([cor_df, pd.Series(data=[round(r, 3), round(p, 3), n], index=["CorrCoef", "pval", "N"])], axis=1)
    cor_df = cor_df.T
    cor_df.index = cols
    cor_df["pval-Bon"] = cor_df["pval"] * len(cols)
    cor_df["abs-cor"] = np.abs(cor_df["CorrCoef"])
    cor_df = cor_df.sort_values(by="abs-cor", ascending=False)
    cor_df["Holm-Bon-Thresh"] = calculate_holm_bon_thresholds(cor_df.loc[:, "pval"], alpha=0.05)
    cor_df["Holm-Bon-k"] = cor_df["pval"] > cor_df["Holm-Bon-Thresh"]
    return cor_df


def calculate_bar_heights_emotions(wdf, emotion_set, action, prefix, pos_key, neg_key, comp_type):
    active_rows = wdf["%sActive" % action] == 1
    for e in emotion_set:
        positive_outcomes = wdf.loc[active_rows, "%s%s%s-%sEvidence-%s" % (prefix, action, pos_key, e, comp_type)]
        negative_outcomes = wdf.loc[active_rows, "%s%s%s-%sEvidence-%s" % (prefix, action, neg_key, e, comp_type)]


def calculate_bar_heights(wdf, emotion, comp_type="DurationProp"):
    scans_pos = wdf.loc[:, "After-ScanPositive-%sEvidence-%s" % (emotion, comp_type)]
    scans_neg = wdf.loc[:, "After-ScanNegative-%sEvidence-%s" % (emotion, comp_type)]

    books_rel = wdf.loc[:, "During-BookRelevant-%sEvidence-%s" % (emotion, comp_type)]
    books_irr = wdf.loc[:, "During-BookIrrelevant-%sEvidence-%s" % (emotion, comp_type)]

    subs_cor = wdf.loc[:, "After-SubmissionCorrect-%sEvidence-%s" % (emotion, comp_type)]
    subs_inc = wdf.loc[:, "After-SubmissionIncorrect-%sEvidence-%s" % (emotion, comp_type)]

    means = np.array([scans_pos.mean(), scans_neg.mean(), books_rel.mean(), books_irr.mean(), subs_cor.mean(), subs_inc.mean()])
    sed = np.sqrt(len(scans_pos))
    std_errors = np.array([scans_pos.std(), scans_neg.std(), books_rel.std(), books_irr.std(), subs_cor.std(), subs_inc.std()]) / sed

    return means, std_errors


def plot_contextual_comparison(wdf, emotion, save_filepath, comp_type="CountRate"):
    """Plot a barchart of the three actions separated by positive and negative outcomes for a specific emotion"""
    fig, ax = plt.subplots(1)

    # Initialize data for bar plot
    x = np.array([0, 1, 5, 6, 10, 11])
    heights, stderrs = calculate_bar_heights(wdf, emotion=emotion, comp_type=comp_type)
    colors = ["gray", "black"] * 3
    tick_labels = ["  After Scan", "", "During Book", "", "After Submission", ""]

    # Plot the bars
    ax.bar(left=x, height=heights, color=colors, tick_label=tick_labels, yerr=stderrs, ecolor="blue")

    # Setting axis labels
    ax.set_ylabel("Average Occurrences in Interval")

    # Creating Legend
    positive = mpatches.Patch(color="gray", label="Positive")
    negative = mpatches.Patch(color="black", label="Negative")
    errors = mpatches.Patch(color="blue", label="Std Error")
    legend_handles = [positive, negative, errors]
    ax.legend(handles=legend_handles, loc=1)

    fig.savefig(save_filepath)

# Data filepath specifications
directory = "C:/Users/robsc/Documents/NC State/GRAWork/Publications/LI-Contextual-Emotions/Data/"
desired_window_filename = directory + "DesiredWindows_Temp.csv"
desired_act_sum_filename = directory + "ActivitySummaryContextEdited.csv"

# Reading in data
wdf = pd.read_csv(desired_window_filename)
adf = pd.read_csv(desired_act_sum_filename)

# Data preprocessing
keep_rows = wdf["TestSubject"].apply(lambda cell: cell in list(adf["TestSubject"]))
wdf = wdf.loc[keep_rows, :]
wdf.index = list(range(wdf.shape[0]))
wdf.index = wdf["TestSubject"]
adf.index = adf["TestSubject"]

keep_submission_rows = wdf["TestSubject"].apply(lambda cell: cell in adf.index and
                                                             adf.loc[cell, "MysterySolved"] and
                                                             adf.loc[cell, "TotalWorksheetSubmits"] > 1)
swdf = wdf.loc[keep_submission_rows, :]
submission_columns = [e for e in wdf.columns if "Submission" in e]
wdf["ScanActive"] = pd.Series((adf.loc[:, "ScanPositive"] > 0) & (adf.loc[:, "ScanNegative"] > 0), dtype=int)
wdf["BookActive"] = pd.Series((adf["BookRelevant"] > 0) & (adf["BookRelevant"] > 0), dtype=int)
wdf["SubmissionActive"] = pd.Series((adf["SubmissionCorrect"] > 0) & (adf["SubmissionIncorrect"] > 0), dtype=int)
wdf.to_csv(directory + "DesiredWindows_Reduced.csv", index=False)

ddf = calculate_repeated_measure_differences(wdf, reduced=True, comp_type="DurationProp")
ddf["ScanActive"] = pd.Series((adf.loc[:, "ScanPositive"] > 0) & (adf.loc[:, "ScanNegative"] > 0), dtype=int)
ddf["BookActive"] = pd.Series((adf["BookRelevant"] > 0) & (adf["BookRelevant"] > 0), dtype=int)
ddf["SubmissionActive"] = pd.Series((adf["SubmissionCorrect"] > 0) & (adf["SubmissionIncorrect"] > 0), dtype=int)

submit_ddf = calculate_repeated_measure_differences(swdf, reduced=True, comp_type="DurationProp")

# Printing the data typically reported in the Participants/Methods section
# Section 2.2
print_demographics(wdf, adf)

# Performing basic visualizations for Section 2.3
fig, ax = plt.subplots(1)
sns.distplot(adf["FinalGameScore"], rug=True, ax=ax)
ax.set_ylabel("Density")
ax.set_xlabel("Overall Game Score")

# Performing action comparison for Section 2.3
adf.loc[wdf["ScanActive"] == 1, ["ScanPositive", "ScanNegative"]].mean(axis=0)
adf.loc[wdf["ScanActive"] == 1, ["ScanPositive", "ScanNegative"]].std(axis=0)
adf.loc[wdf["BookActive"] == 1, ["BookRelevant", "BookIrrelevant"]].mean(axis=0)
adf.loc[wdf["BookActive"] == 1, ["BookRelevant", "BookIrrelevant"]].std(axis=0)
adf.loc[wdf["SubmissionActive"] == 1, ["SubmissionCorrect", "SubmissionIncorrect"]].mean(axis=0)
adf.loc[wdf["SubmissionActive"] == 1, ["SubmissionCorrect", "SubmissionIncorrect"]].std(axis=0)
wdf.loc[:, ["ScanActive", "BookActive", "SubmissionActive"]].sum()
# Plot the significant emotions bar charts
# Back part of RQ1
image_directory = "C:/Users/robsc/Documents/NC State/GRAWork/Publications/LI-Contextual-Emotions/Images/"
plot_contextual_comparison(wdf, emotion="Surprise", save_filepath=image_directory+"SurpriseBar.png")
plot_contextual_comparison(wdf, emotion="Fear", save_filepath=image_directory+"FearBar.png")

# Part of justifying Final Game Score
ss.pearsonr(adf.loc[:, "FinalGameScore"], adf.loc[:, "RevisedNLG"])
ss.pearsonr(adf.loc[:, "FinalGameScore"], adf.loc[:, "RevisedPost"])

# RQ1 results come from HotellingTest.R script

# Getting correlations between differences of emotions and learning outcomes
# Results RQ2

gs_cor_tab = calculate_correlation_table(ddf, adf, "FinalGameScore")
print("Final Game Score Correlation Table")
print(gs_cor_tab.head())
srows = ddf["ScanActive"] == 1
sns.distplot(ddf.loc[srows, "After-Scan-Confusion-Difference"])
sns.regplot(x=ddf.loc[srows, "During-Book-Joy-Difference"], y=adf.loc[srows, "FinalGameScore"])
nlg_cor_tab = calculate_correlation_table(ddf, adf, "RevisedNLG")
print("NLG Correlation Table")
print(nlg_cor_tab.head())
lg_cor_tab = calculate_correlation_table(ddf, adf, "RevisedLG")
print("LG Correlation Table")
print(lg_cor_tab.head())
dur_cor_tab = calculate_correlation_table(ddf, adf, "Duration")
print("Duration Correlation Table")
print(dur_cor_tab.head())

# RQ3 predictive modeling using the differences
response = "FinalGameScore"
context_feats = [e for e in ddf.columns if ("Joy" in e or "Confusion" in e or "Frustration" in e) and "Submission" not in e]
action_feats = ["TotalWorksheetSubmits"]
ddf.loc[:, context_feats].corr()
X = pd.concat([ddf.loc[:,context_feats], adf.loc[:, action_feats]], axis=1)
Xc = sm.add_constant(X)
y = adf.loc[:, response].values.reshape(-1, 1)
results = sm.OLS(y, Xc).fit()
print(results.summary())

# To get standardized Betas
scaler = StandardScaler()
Xscale = scaler.fit_transform(X)
Xscalec = sm.add_constant(Xscale)
yscale = scaler.fit_transform(adf.loc[:, response].values.reshape(-1, 1))
resultsscale = sm.OLS(yscale, Xscalec).fit()
print(resultsscale.summary())

# Evaluate cross-validation R2 of model
errors = []
mean_errors = []
submit_errors = []
kfold = KFold(n_splits=X.shape[0])
for train, test in kfold.split(X):
    model = LinearRegression()
    X_train, X_test = X.iloc[train, :], X.iloc[test, :]
    y_train, y_test = y[train], y[test]
    model.fit(X=X_train, y=y_train)

    model_submit = LinearRegression()
    Xs_train, Xs_test = X["TotalWorksheetSubmits"].iloc[train].values.reshape(-1, 1), X["TotalWorksheetSubmits"].iloc[test].values.reshape(-1, 1)
    model_submit.fit(X=Xs_train, y=y_train)

    mean_errors.append((y_test[0] - y_train.mean())[0])

    prediction = model.predict(X=X_test)[0]
    errors.append((y_test[0] - prediction)[0])

    prediction_s = model_submit.predict(X=Xs_test)[0]
    submit_errors.append((y_test[0] - prediction_s)[0])

model = LinearRegression()
model.fit(X=X, y=y)
sse = np.sum((y - model.predict(X=X))**2)

model = LinearRegression()
model.fit(X=X.loc[:, "TotalWorksheetSubmits"].values.reshape(-1, 1), y=y)
sses = np.sum((y - model.predict(X=X.loc[:, "TotalWorksheetSubmits"].values.reshape(-1, 1)))**2)

rmse = np.mean(np.array(errors)**2)
rsmse = np.mean(np.array(submit_errors)**2)
tmse = np.mean(np.array(mean_errors)**2)

cvr2 = 1 - rmse / tmse
cvrs2 = 1 - rsmse / tmse
cvrc2 = 1 - rmse / rsmse

pct_improve = (rsmse - rmse) / rsmse

F_num = ((sses - sse)) / 6
F_denom = ((sse) / ((61 - 8)))
F_stat = F_num / F_denom
df1 = 6
df2 = (61 - 8)
print(1 - ss.f.cdf(F_stat, df1, df2))
