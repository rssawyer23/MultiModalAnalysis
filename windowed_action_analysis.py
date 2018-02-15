from scipy.stats import ttest_ind
import pandas as pd
import numpy as np

windowed_action_filename = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/EventSequence/WindowedActionsAll.csv"
condition_output_filename = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/EventSequence/WindowedActionsAll-ConditionAnalysis.csv"
action_output_filename = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/EventSequence/WindowedActionsAll-ActionAnalysis.csv"


def perform_condition_analysis(data_filename, output_filename):
    with open(output_filename, 'w') as ofile:
        ofile.write("ColumnCompared,MeanFull,StdFull,MeanPartial,StdPartial,t,p\n")
        data = pd.read_csv(data_filename)
        col_names = list(data.columns.values)[1:]
        full_rows = data.apply(lambda row: row["TestSubject"][5] == "1", axis=1)
        partial_rows = data.apply(lambda row: row["TestSubject"][5] == "2", axis=1)
        for col in col_names:
            if data[col].dtype == 'float64':
                nan_rows = np.isnan(data[col]) == False
                tdata = data.loc[nan_rows, :]
                mean_a = tdata.loc[full_rows, col].mean()
                mean_b = tdata.loc[partial_rows, col].mean()
                std_a = tdata.loc[full_rows, col].std()
                std_b = tdata.loc[partial_rows, col].std()
                t, p = ttest_ind(a=tdata.loc[full_rows, col], b=tdata.loc[partial_rows, col], equal_var=False)
                to_write = "%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.5f\n" % (col, mean_a, std_a, mean_b, std_b, t, p)
                ofile.write(to_write)


def get_emotions(columns):
    emotions = []
    for e in columns:
        try:
            emotion = e.split("-")[2]
            if emotion not in emotions:
                emotions.append(emotion)
        except IndexError:
            pass
    return emotions


def perform_action_analysis(windowed_action_filename, output_filename):
    with open(output_filename, 'w') as ofile:
        ofile.write("Action,EmotionCompared,MeanBefore,StdBefore,MeanAfter,StdAfter,t,p\n")
        data = pd.read_csv(windowed_action_filename)
        col_names = list(data.columns.values)[1:]
        actions = [e[:-6] for e in col_names if "-Count" in e and "-CountRate" not in e]
        emotions = get_emotions(col_names)
        for action in actions:
            for emotion in emotions:
                mean_a = data.loc[:, "Before-%s-%s-CountRate" % (action, emotion)].mean()
                mean_b = data.loc[:, "After-%s-%s-CountRate" % (action, emotion)].mean()
                std_a = data.loc[:, "Before-%s-%s-CountRate" % (action, emotion)].std()
                std_b = data.loc[:, "After-%s-%s-CountRate" % (action, emotion)].std()
                t, p = ttest_ind(a=data.loc[:, "Before-%s-%s-CountRate" % (action, emotion)], b=data.loc[:, "After-%s-%s-CountRate" % (action, emotion)], equal_var=False)
                to_write = "%s,%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.5f\n" % (action, emotion, mean_a, std_a, mean_b, std_b, t, p)
                ofile.write(to_write)


if __name__ == "__main__":
    #perform_condition_analysis(windowed_action_filename, condition_output_filename)
    perform_action_analysis(windowed_action_filename, action_output_filename)
