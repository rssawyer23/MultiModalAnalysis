import pandas as pd


def add_standardized_columns(trial_summary_appended, trial_summary_standardized, mj_cols):
    """
    Function for adding a student standardized column for each metacognitive judgment measure
    :param trial_summary_appended: String for filename of the DataFrame containing mj_cols
    :param trial_summary_standardized: String for filename of output after adding standardized columns
    :param mj_cols: list of columns that should be standardized by student
    :return: None (write to trial_summary_standardized)
    """
    df = pd.read_csv(trial_summary_appended)
    for col in mj_cols:
        df[col+"-Std"] = pd.Series([0] * df.shape[0])
        df[col+"-Real"] = pd.Series([0] * df.shape[0])
    subjects = df["TestSubject"].unique()
    for subject in subjects:
        subject_rows = df["TestSubject"] == subject
        # Standardize the values using only the test subjects' judgements
        for col in mj_cols:
            subject_avg = df.loc[subject_rows, col].mean()
            subject_std = df.loc[subject_rows, col].std()
            df.loc[subject_rows, col+"-Std"] = (df.loc[subject_rows, col] - subject_avg) / subject_std
            start_index = df.loc[subject_rows, col].index[0]

            #Use average and standard deviation up to the specific trial (realistic running averages)
            for i in df.loc[subject_rows, col].index:
                subject_avg = df.loc[start_index:i, col].mean()
                subject_std = df.loc[start_index:i, col].std()
                if (i - start_index) == 0:
                    df.loc[i, col+"-Real"] = 0
                else:
                    df.loc[i, col + "-Real"] = (df.loc[i, col] - subject_avg) / subject_std

    df.to_csv(trial_summary_standardized, index=False)


def get_ease(other_rows, desired_content):
    """Use content ids from students other than current student to determine question ease"""
    content_rows = other_rows["ContentIdNum"] == desired_content
    qe = other_rows.loc[content_rows, "MultipleChoiceScores"].sum()
    all_mean = other_rows.groupby("ContentIdNum")["MultipleChoiceScores"].sum().mean()
    all_std = other_rows.groupby("ContentIdNum")["MultipleChoiceScores"].sum().std()
    std_qe = (qe - all_mean) / all_std
    return std_qe


def add_mc_ease(appended_filename, ease_output):
    df = pd.read_csv(appended_filename)
    subjects = df["TestSubject"].unique()
    df["MC-Ease"] = pd.Series([0] * df.shape[0])
    for subject in subjects:
        subject_rows = df["TestSubject"] == subject

        for i in df.loc[subject_rows, :].index:
            q_ease = get_ease(df.loc[~subject_rows, ["MultipleChoiceScores", "ContentIdNum"]], df.loc[i, "ContentIdNum"])
            df.loc[i, "MC-Ease"] = q_ease
    df.to_csv(ease_output, index=False)


def add_eol_categories(appended_filename, category_output, mj_cols):
    """Add the category of the student based on EOL splits (above/below 0, ease centered above/below 0)"""
    df = pd.read_csv(appended_filename)
    for col in mj_cols:
        df[col+"-Category"] = pd.Series(df[col+"-Std"] > 0, dtype=int)
    df["EOL-Centered-Category"] = pd.Series(df["EaseOfLearning-Std"] - df["MC-Ease"] > 0, dtype=int)
    df.to_csv(category_output, index=False)


if __name__ == "__main__":
    fn = "C:/Users/robsc/Documents/NC State/GRAWork/IVHData/OutputJan18Neg/FACET-ThresholdCrossed/FACET-Sequence-Summary-Appended.csv"
    o_fn = "C:/Users/robsc/Documents/NC State/GRAWork/IVHData/OutputJan18Neg/FACET-ThresholdCrossed/FACET-Sequence-Summary-Appended-Std.csv"
    d_fn = "C:/Users/robsc/Documents/NC State/GRAWork/IVHData/OutputJan18Neg/FACET-ThresholdCrossed/FACET-Sequence-Summary-Appended-Diff.csv"
    c_fn = "C:/Users/robsc/Documents/NC State/GRAWork/IVHData/OutputJan18Neg/FACET-ThresholdCrossed/FACET-Sequence-Summary-Appended-Categories.csv"
    add_standardized_columns(fn, o_fn, ["MultipleChoiceConfidence", "EaseOfLearning", "JustificationConfidence"])
    add_mc_ease(o_fn, d_fn)
    add_eol_categories(d_fn, c_fn, ["MultipleChoiceConfidence", "EaseOfLearning", "JustificationConfidence"])
