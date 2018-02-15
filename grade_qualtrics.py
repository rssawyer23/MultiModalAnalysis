import pandas as pd
import numpy as np

grading_key_filename = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/SurveyGradingScheme.csv"
post_test_filename = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/CI_PostTest_Data_8-31-17.csv"
pre_test_filename = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/CI_PreTest_Data_8-31-17.csv"
activity_summary_filename = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/ActivitySummary/ActivitySummary.csv"
output_filename = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/ActivitySummary/ActivitySummaryGraded.csv"
normalize = True
show = True
INVALID_QUESTIONS = []


def nlg(row, max_score=21):
    pre = row["PreTestScore"]
    post = row["PostTestScore"]
    if np.isnan(row["PreTestScore"]) or np.isnan(row["PostTestScore"]) or row["PreTestScore"] == 0:
        return 0
    if post >= pre:
        return (post - pre) / (max_score - pre)
    else:
        return (post - pre) / pre


def get_survey_names(key_df):
    surveys = list(key_df["SurveyName"].unique())
    for survey in surveys:
        survey_rows = key_df["SurveyName"] == survey
        subscales = list(key_df.loc[survey_rows, "Subscale"].unique())
        if "Exploratory" in subscales:
            subscales.remove("Exploratory")
        if "Base" in subscales:
            subscales.remove("Base")
        for sub in subscales:
            surveys.append("%s-%s" % (survey, sub))
    return surveys


def grade(student_row, key, survey_names, normalize=True, pre=False):
    # Initialize the grades and counts for each survey and subscale survey
    grades = dict()
    q_counts = dict()

    if pre:
        prefix = "Pre-"
    else:
        prefix = "Post-"

    for s in survey_names:
        grades[prefix+s] = 0
        q_counts[prefix+s] = 0

    for index, row in key.iterrows():
        try:
            score = student_row[row["QuestionID"]]
            if score < row["LowRange"]:  # Ensure score within bounds, otherwise set to cap
                score = row["LowRange"]
            elif score > row["HighRange"]:
                score = row["HighRange"]
            if row["Reverse"]:  # A reverse questions should be graded as if high = low
                score = row["HighRange"] - score + row["LowRange"]

            if row["Subscale"] != "Exploratory":  # Do not include exploratory questions (for testing new questions)
                grades[prefix+row["SurveyName"]] += score
                q_counts[prefix+row["SurveyName"]] += 1
                if row["Subscale"] != "Base":  # If belongs to subscale, add score to that subscale
                    grades["%s%s-%s" % (prefix, row["SurveyName"],row["Subscale"])] += score
                    q_counts["%s%s-%s" % (prefix, row["SurveyName"],row["Subscale"])] += 1
        except KeyError:
            if row["QuestionID"] not in INVALID_QUESTIONS:
                INVALID_QUESTIONS.append(row["QuestionID"])

    if normalize:  # Divide by the counts to get an average score if flag is true
        for k in grades.keys():
            try:
                grades[k] = grades[k] / float(q_counts[k])
            except ZeroDivisionError:
                grades[k] = 0
    return grades


def get_grade_df(survey_df, key, survey_names, normalize=False, pre=False):
    student_grades = dict()
    for index, row in survey_df.iterrows():
        grades = grade(row, key, survey_names, normalize=normalize, pre=pre)
        student_grades[row["username"]] = grades
    grade_df = pd.DataFrame.from_dict(student_grades, orient="index")

    # Removing surveys that were unable to be graded by checking if all 0s
    del_cols = []
    for col in list(grade_df.columns.values):
        if np.all(grade_df[col] == 0):
            del_cols.append(col)
    if show:
        print("Ungraded surveys: %s" % del_cols)
    grade_df.drop(del_cols, axis=1, inplace=True)
    grade_df["TestSubject"] = grade_df.index
    grade_df.index = list(range(grade_df.shape[0]))
    return grade_df


def grade_tests(grading_key_filename, post_test_filename, pre_test_filename, activity_summary_filename, output_filename, normalize=True, show=False):
    key = pd.read_csv(grading_key_filename)
    post = pd.read_csv(post_test_filename)
    pre = pd.read_csv(pre_test_filename)
    survey_names = get_survey_names(key_df=key)

    post_grade_df = get_grade_df(post, key, survey_names, normalize=normalize, pre=False)
    pre_grade_df = get_grade_df(pre, key, survey_names, normalize=normalize, pre=True)

    act_sum = pd.read_csv(activity_summary_filename, usecols=lambda colname: type(colname) is str and "VideoGamesPlayed" not in colname)
    act_sum["NLG"] = act_sum.apply(nlg, axis=1)
    act_sum["Condition"] = act_sum.apply(lambda row: row["TestSubject"][5], axis=1)

    joined_df = act_sum.merge(right=post_grade_df, how='left', on="TestSubject")
    joined_df = joined_df.merge(right=pre_grade_df, how='left', on='TestSubject')
    joined_df.to_csv(output_filename, index=False)

if __name__ == "__main__":
    grade_tests(grading_key_filename, post_test_filename, pre_test_filename, activity_summary_filename, output_filename,
                    normalize=normalize, show=show)
