# Script for plotting duration versus key time series characteristics for EDM2018 paper on trajectories
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dateutil.parser import parse


def revised_durations(data, omit_list):
    subjects = [e for e in list(data.loc[:, "TestSubject"].unique()) if e not in omit_list and pd.notna(e)]
    duration_series = pd.Series(index=subjects)
    for subject in subjects:
        subject_rows = data.loc[:, "TestSubject"] == subject
        subject_data = data.loc[subject_rows, :]
        subject_start_time = parse(subject_data.loc[:, "TimeStamp"].iloc[0])
        subject_durations = subject_data.loc[:, "TimeStamp"].apply(lambda cell: (parse(cell) - subject_start_time).total_seconds())
        revised_duration = subject_durations.iloc[-1] - subject_durations.iloc[0]
        duration_series[subject] = revised_duration
        print(subject_data.columns)
    return duration_series

#event_input = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/EventSequence/EventSequence_wCGS_NoAOI.csv"
input_filename = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/ActivitySummary/ActivitySummaryCumulativesRevised.csv"
#event_df = pd.read_csv(event_input)
#student_durations = revised_durations(event_df, omit_list=[])
#s_df = pd.DataFrame(student_durations, columns=["RevisedDuration"])
#s_df["TestSubject"] = s_df.index

s_df = pd.read_csv("C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/Intermediate/MetricsOutput.csv")
s_df["TestSubject"] = s_df.iloc[:, 0]

df = pd.read_csv(input_filename)
df = df.merge(right=s_df, how='left', on="TestSubject")
df.to_csv("C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/ActivitySummary/ActivitySummaryCumulativesRevisedPhases.csv", index=False)
#df["Color"] = df["Condition"].apply(lambda cell: tuple([0.5,0.5,0.5]) if cell == 1 else tuple([0.0,0.0,0.0]))

full_agency_rows = df["Condition"] == 1.0
df = df.loc[full_agency_rows, :]

df.loc[:, "DurationDist"] = df.loc[:, "RevisedDuration"] - 5464.0

fig, ax = plt.subplots(1)
ax.set_title("Full Agency Student Duration vs Distance")
ax.set_xlabel("Total Gameplay Duration")
ax.set_ylabel("Golden Path Distance")
ax.scatter(x=np.abs(df["RevisedDuration"]), y=df["Average"], s=3.0, label='Students')
ax.axvline(x=5464, linestyle='dotted', color='r', label='Gold Path Duration')
ax.legend()
fig.savefig("C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/ActivitySummary/Plots/DurationScatterFullRevised.png")
#plt.show()

