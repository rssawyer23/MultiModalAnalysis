import pandas as pd
import numpy as np


def add_revised_nlg(row):
    if row["RevisedLG"] >= 0:
        nlg = row["RevisedLG"] / (21 - row["RevisedScore"])
    else:
        nlg = row["RevisedLG"] / (row["RevisedScore"])
    return nlg


def add_nlg(row):
    if row["LG"] >= 0:
        nlg = row["LG"] / (21 - row["Score-sum"])
    else:
        nlg = row["LG"] / row["Score-sum"]
    return nlg


def add_revised_nlg_df(base_act_sum, revised_act_sum, post_survey_fn, pre_survey_fn, subject_col="username"):
    # Post test rescoring
    postdf = pd.read_csv(post_survey_fn)
    postdf["RevisedScore"] = postdf["Score-sum"]
    postdf["ScoreModifier"] = np.zeros(postdf.shape[0])

    # Add one for the actual correct answer, subtract one for the incorrectly defined correct answer
    postdf["ScoreModifier"] += pd.Series(postdf["Q30"] == 4, dtype=int)
    postdf["ScoreModifier"] -= pd.Series(postdf["Q30"] == 1, dtype=int)

    postdf["ScoreModifier"] += pd.Series(postdf["Q26"] == 3, dtype=int)
    postdf["ScoreModifier"] -= pd.Series(postdf["Q26"] == 1, dtype=int)

    postdf["ScoreModifier"] += pd.Series(postdf["Q24"] == 1, dtype=int)
    postdf["ScoreModifier"] -= pd.Series(postdf["Q24"] == 3, dtype=int)

    # Question has two correct answers, add back in for the ones falsely penalized
    postdf["ScoreModifier"] += pd.Series(postdf["Q21"] == 3, dtype=int)

    print("Post Test Score Modifier Descriptions")
    print(postdf.loc[:, ["ScoreModifier", "RevisedScore", "Score-sum"]].describe())

    postdf["RevisedScore"] += postdf["ScoreModifier"]

    # Pre test rescoring
    predf = pd.read_csv(pre_survey_fn)

    predf["RevisedScore"] = predf["Score-sum"]
    predf["ScoreModifier"] = np.zeros(predf.shape[0])

    predf["ScoreModifier"] += pd.Series(predf["Q31"] == 2, dtype=int)

    predf["RevisedScore"] += predf["ScoreModifier"]

    print("Pre Test Score Modifier Description")
    print(predf.loc[:, ["ScoreModifier", "RevisedScore", "Score-sum"]].describe())

    # Learning Gain Rescoring
    predf["LG"] = postdf["Score-sum"] - predf["Score-sum"]
    predf["RevisedLG"] = postdf["RevisedScore"] - predf["RevisedScore"]
    print(predf.loc[:, ["LG", "RevisedLG"]].describe())

    predf["RevisedNLG"] = predf.apply(add_revised_nlg, axis=1)
    predf["NLG"] = predf.apply(add_nlg, axis=1)
    predf["NLG-Diff"] = predf["RevisedNLG"] - predf["NLG"]

    predf["RevisedPre"] = predf["RevisedScore"]
    predf["RevisedPost"] = postdf["RevisedScore"]

    act_df = pd.read_csv(act_summary_fn)
    act_df.index = act_df["TestSubject"]
    predf.index = predf["username"]

    rev_act_df = act_df.join(predf.loc[:, ["RevisedPre", "RevisedPost", "RevisedLG", "RevisedNLG"]])
    rev_act_df.to_csv(rev_summary_fn, index=False)


if __name__ == "__main__":
    post_survey_fn = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/CI_PostTest_Data 3-1-18.csv"
    act_summary_fn = 'C:/Users/robsc/Documents/NC State/GRAWork/CIData/OutputFull2018/ActivitySummary/ActivitySummaryAppended-All.csv'
    rev_summary_fn = 'C:/Users/robsc/Documents/NC State/GRAWork/CIData/OutputFull2018/ActivitySummary/ActivitySummaryAppendedRevised-All.csv'
    pre_survey_fn = 'C:/Users/robsc/Documents/NC State/GRAWork/CIData/CI_PreTest_Data 3-1-18.csv'
    subject_col = "username"

    add_revised_nlg_df(base_act_sum=act_summary_fn,
                       revised_act_sum=rev_summary_fn,
                       post_survey_fn=post_survey_fn,
                       pre_survey_fn=pre_survey_fn,
                       subject_col=subject_col)



# print(predf.loc[:, ["RevisedNLG", "NLG", "NLG-Diff"]].describe())
#
# full_agency_rows = predf["username"].apply(lambda x: "CI1301" in x)
# print(predf.loc[full_agency_rows, ["RevisedNLG", "NLG", "NLG-Diff"]].describe())
#
# partial_agency_rows = predf['username'].apply(lambda x: "CI1302" in x)
# print(predf.loc[partial_agency_rows, ["RevisedNLG", "NLG", "NLG-Diff"]].describe())
#
# no_agency_rows = predf["username"].apply(lambda x: "CI1303" in x)
# print(predf.loc[no_agency_rows, ["RevisedNLG", "NLG", "NLG-Diff"]].describe())