# Script for creating a FACET-Event file from the raw facet file for an absolute thresholded

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from matplotlib import colors as colors
from dateutil.parser import parse
import os
import datetime


def student_evidence_summary(student_data, emotion_cols):
    """Gets means and standard deviations of baseline corrected evidence scores for one student"""
    means = student_data.loc[:, emotion_cols].mean()
    sds = student_data.loc[:, emotion_cols].std()
    series_dict = dict()
    series_ind = []
    for e in emotion_cols:
        series_ind.append("%s-Mean" % e)
        series_ind.append("%s-Std" % e)
        series_dict["%s-Mean" % e] = means[e]
        series_dict["%s-Std" % e] = sds[e]
    return pd.Series(series_dict, index=series_ind)


def _check_directory(dir):
    """Ensure directory exists, otherwise make directory"""
    if not os.path.isdir(dir):
        os.makedirs(dir)


def save_plot(s_data, emotions, facet_events, save_directory, reduce_index=10):
    """Output plots of evidence scores given by emotions for one student"""
    _check_directory(save_directory)

    student_id = s_data["TestSubject"].unique()[0]
    student_start = parse(s_data.loc[:,"TimeStamp"].iloc[0])
    #s_data.index = pd.to_datetime(s_data.loc[:, "TimeStamp"])
    s_data.loc[:, "DurationElapsed"] = s_data["TimeStamp"].apply(lambda cell: (parse(cell) - student_start).total_seconds())

    all_colors = ["b", "g", "c", "m", "y", "k", "w", "r"]
    emo_colors = all_colors[:len(emotions)]

    emotion_rows = s_data.loc[:, ["DurationElapsed", "ConfusionEvidence"]]

    fig, ax = plt.subplots(1)
    ax.set_xlim(left=0, right=emotion_rows["DurationElapsed"].iloc[-1])
    ax.set_ylim(bottom=-6.0, top=6.0)

    for col, emotion in zip(emo_colors, emotions):
        emotion_rows = s_data.loc[:, ["DurationElapsed", emotion]]
        a = np.array(emotion_rows.loc[:, "DurationElapsed"])
        b = np.array(emotion_rows.loc[:, emotion])
        ax.plot(a, b, "%s-" % col)

        keep_rows = facet_events["Target"].apply(lambda cell: cell == emotion)
        facet_reduced = facet_events.loc[keep_rows, :]
        facet_reduced.loc[:, "DurationElapsed"] = facet_reduced["TimeStamp"].apply(lambda cell: (parse(cell) - student_start).total_seconds())
        facet_reduced.index = list(range(facet_reduced.shape[0]))
        facet_reduced.loc[:, "Upper"] = pd.Series([5.0]*facet_reduced.shape[0])
        if facet_reduced.shape[0] > 0:
            ax.scatter(x=np.array(facet_reduced.loc[:,"DurationElapsed"]), y=np.array(facet_reduced.loc[:,"Upper"]),
                     color=col, marker="o", label=emotion.replace("Evidence", " Event"), s=np.array(facet_reduced.loc[:,"Duration"])*2.0)
    ax.set_xlabel("Duration Elapsed")
    ax.set_ylabel("FACET Evidence")
    ax.set_title(student_id)
    ax.legend(emotions, loc="lower right")
    fig.savefig(save_directory + "/" + student_id + ".png")
    plt.close()


def print_time(start_time, keyword):
    current_time = datetime.datetime.now()
    print("Finished %s: %s (%.4f seconds)" % (keyword, current_time, (current_time-start_time).total_seconds()))


def output_FACET_evidence(target_dir, data_extension, facet_event_filename, emotions_to_use, output_diagnostics=True, output_correlations=True, output_figures=True, include_aus=True, merged=True):
    """Gets means and standard deviations of the baseline corrected evidence scores for a students full gameplay"""
    start = datetime.datetime.now()
    print("Starting FACET Diagnostics: %s" % start)
    emotions = [e+"Evidence" for e in emotions_to_use]

    facet_events = pd.read_csv(facet_event_filename)
    print_time(start, "Reading FACET-Events")

    if merged:
        student_data = pd.read_csv(target_dir + data_extension)
        if include_aus:
            emotion_cols = [e for e in student_data.columns.values if "Evidence" in e and "Raw" not in e]
        else:
            emotion_cols = [e for e in student_data.columns.values if "Evidence" in e and "AU" not in e and "Raw" not in e]
        print_time(start, "Reading Data")

        if output_correlations:
            student_data.loc[:, emotion_cols].corr().to_csv(target_dir+"/FACET-Correlations.csv", index=True)
            print_time(start, "FACET-Correlations")

        student_ids = list(student_data["TestSubject"].unique())
        student_df = pd.DataFrame()
        for student_id in student_ids:
            student_rows = student_data["TestSubject"] == student_id
            specific_student_data = student_data.loc[student_rows, :]
            facet_rows = facet_events["TestSubject"] == student_id
            facet_student_events = facet_events.loc[facet_rows, :]
            student_series = student_evidence_summary(specific_student_data, emotion_cols)
            if output_figures:
                save_plot(specific_student_data,
                          emotions=emotions,
                          facet_events=facet_student_events,
                          save_directory=target_dir+"/FACET-Plots",
                          reduce_index=1)
            student_df = pd.concat([student_df, student_series], axis=1)
    else:  # FOR UNMERGED FACET FILES, I.E. EACH STUDENT HAS INDIVIDUAL FACET FILE
        student_files = [e for e in os.listdir(target_dir) if "_PN" in e]
        student_df = pd.DataFrame()
        student_ids = []
        for student_file in student_files:
            specific_student_data = pd.read_csv(target_dir + "/" + student_file)
            student_id = student_file.split("-")[0]
            student_ids.append(student_id)
            facet_rows = facet_events["TestSubject"] == student_id
            facet_student_events = facet_events.loc[facet_rows, :]
            if include_aus:
                emotion_cols = [e for e in specific_student_data.columns.values if "Evidence" in e and "Raw" not in e]
            else:
                emotion_cols = [e for e in specific_student_data.columns.values if "Evidence" in e and "AU" not in e and "Raw" not in e]
            student_series = student_evidence_summary(specific_student_data, emotion_cols)
            if output_figures:
                save_plot(specific_student_data,
                          emotions=emotions,
                          facet_events=facet_student_events,
                          save_directory=target_dir+"/FACET-Plots",
                          reduce_index=1)
            student_df = pd.concat([student_df, student_series], axis=1)
            print_time(start, student_file)
    print_time(start, "FACET-Plots")

    if output_diagnostics:
        student_df.columns = student_ids
        student_df.to_csv(target_dir+"/FACET-Diagnostics.csv", index=True)
        print_time(start, "FACET-Diagnostics")

    return student_df


if __name__ == "__main__":
    facet_type = "/FACET-Z-Score" # works with "/FACET" or "/FACET-Z-Score"
    target_directory = "C:/Users/robsc/Documents/NC State/GRAWork/IVHData/OutputJan18Neg%s" % facet_type
    facet_event_filename = "C:/Users/robsc/Documents/NC State/GRAWork/IVHData/OutputJan18Neg/FACET-ThresholdCrossed/FACET-Events.csv"  # GENERATED FROM FACET_EVENT_FILE.PY
    merged = True
    student_df = output_FACET_evidence(target_dir=target_directory,
                                       data_extension="%s.csv" % facet_type,
                                       facet_event_filename=facet_event_filename,
                                       output_diagnostics=False,
                                       output_correlations=False,
                                       output_figures=True,
                                       emotions_to_use=["Confusion", "Joy"],
                                       include_aus=True,
                                       merged=merged)
    print(student_df)

