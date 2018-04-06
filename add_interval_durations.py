import pandas as pd
import numpy as np
event_sequence_filename = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/EventSequence/EventSequence.csv"
activity_summary_filename = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/ActivitySummary/ActivitySummaryGraded.csv"
activity_interval_filename = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/ActivitySummary/ActivitySummaryGradedIntervals.csv"

interval_names = ["Tutorial", "PreScan", "PostScan"]
interval_plots = ["TutorialComplete", "TestObject", "SolvedMystery"]


def _get_tutorial_df(event_df):
    subjects = list(event_df["TestSubject"].unique())
    non_aoi = event_df["Event"] != "AOI"
    event_df = event_df.loc[non_aoi, :]
    event_df.index = list(range(event_df.shape[0]))

    subject_list = []
    for subject in subjects:
        subject_rows = event_df["TestSubject"] == subject
        subject_df = event_df.loc[subject_rows, :]
        subject_df.index = list(range(subject_df.shape[0]))
        non_tutorial_rows = subject_df["Location"].apply(lambda cell: str(cell) != "" and str(cell) != "Dock" and "Beach" not in str(cell))
        subject_non_tutorial_df = subject_df.loc[non_tutorial_rows, :]
        try:
            game_time = subject_non_tutorial_df["GameTime"].iloc[0]
            subject_list.append([subject, game_time])
        except IndexError:
            print("Error finding tutorial duration of %s" % subject)
    tutorial_df = pd.DataFrame(subject_list, columns=["TestSubject", "Duration-Tutorial"])
    tutorial_df.index = tutorial_df["TestSubject"]
    return tutorial_df



def add_interval_durations(event_sequence_filename, activity_summary_filename, output_filename):
    event_df = pd.read_csv(event_sequence_filename)
    act_sum_df = pd.read_csv(activity_summary_filename)

    duration_df = act_sum_df.loc[:,["TestSubject"]]

    tutorial_df = _get_tutorial_df(event_df)

    # Deprecated since TutorialComplete point is now last time student originally on beach
    # tutorial_rows = event_df["Name"] == "TutorialComplete"
    # tutorial_df = event_df.loc[tutorial_rows, ["TestSubject", "GameTime"]]
    # tutorial_df.columns = ["TestSubject", "Duration-Tutorial"]
    # tutorial_df.index = tutorial_df["TestSubject"]

    pre_scan_rows = event_df["Name"] == "TestObject"
    pre_scan_df = event_df.loc[pre_scan_rows, ["TestSubject", "GameTime"]]
    pre_scan_df.columns = ["TestSubject", "Duration-PreScan"]
    pre_scan_df.index = pre_scan_df["TestSubject"]
    pre_scan_df["Duration-PreScan"] -= tutorial_df["Duration-Tutorial"]
    negative_rows = pre_scan_df["Duration-PreScan"] < 0
    pre_scan_df.loc[negative_rows,"Duration-PreScan"] = 0

    post_scan_df = act_sum_df.loc[:,["TestSubject", "Duration"]]
    post_scan_df.index = post_scan_df["TestSubject"]
    post_scan_df["Duration"] = post_scan_df["Duration"] - tutorial_df["Duration-Tutorial"] - pre_scan_df["Duration-PreScan"]
    post_scan_df.columns = ["TestSubject", "Duration-PostScan"]

    act_sum_df = act_sum_df.merge(right=tutorial_df,how='left',on='TestSubject')
    act_sum_df = act_sum_df.merge(right=pre_scan_df,how='left',on='TestSubject')
    act_sum_df = act_sum_df.merge(right=post_scan_df,how='left',on='TestSubject')

    act_sum_df.to_csv(output_filename, index=False)

if __name__ == "__main__":
    add_interval_durations(event_sequence_filename=event_sequence_filename,
                           activity_summary_filename=activity_summary_filename,
                           output_filename=activity_interval_filename)
