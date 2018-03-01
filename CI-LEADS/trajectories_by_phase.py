import numpy as np
import pandas as pd
import trajectories_crystal_island as tci
from dateutil.parser import parse
from sklearn.linear_model import LinearRegression


def _get_tutorial_timestamp(subject_df):
    """Given one subject's event dataframe, return the tutorial timestamp and game duration played"""
    start_timestamp = parse(subject_df["TimeStamp"].iloc[0])
    elise_beach_rows = subject_df["NPC"] == "Elise Beach"
    elise_df = subject_df.loc[elise_beach_rows, :]
    tutorial_timestamp = parse(elise_df["TimeStamp"].iloc[-1])
    return tutorial_timestamp, (tutorial_timestamp - start_timestamp).total_seconds()


def _get_first_scan_timestamp(subject_df):
    """Given one subject's event dataframe, return the first scan timestamp and duration between start and scan"""
    start_timestamp = parse(subject_df["TimeStamp"].iloc[0])
    scan_rows = subject_df["Event"] == "Scanner"
    scan_df = subject_df.loc[scan_rows, :]
    scan_timestamp = parse(scan_df["TimeStamp"].iloc[0])
    return scan_timestamp, (scan_timestamp - start_timestamp).total_seconds()


def _get_post_scan_time(subject_df):
    """Given one subject's event dataframe, return the final timestamp and duration between first scan and completion"""
    start_timestamp = parse(subject_df["TimeStamp"].iloc[0])
    end_timestamp = parse(subject_df["TimeStamp"].iloc[-1])
    return end_timestamp, (end_timestamp - start_timestamp).total_seconds()


def get_tutorial_df(subject_df):
    elise_beach_rows = subject_df["NPC"] == "Elise Beach"
    elise_df = subject_df.loc[elise_beach_rows, :]
    final_index = elise_df.index[-1]
    tutorial_df = subject_df.iloc[:final_index, :]
    return tutorial_df, final_index


def get_prescan_df(subject_df, prev_index):
    scan_rows = subject_df["Event"] == "Scanner"
    scan_df = subject_df.loc[scan_rows, :]
    final_index = scan_df.index[-1]
    prescan_df = subject_df.iloc[prev_index:final_index, :]
    return prescan_df, final_index


def get_postscan_df(subject_df, prev_index):
    return subject_df.iloc[prev_index:, :]


def get_subject_phase_scatter(subject_data, x="DurationElapsed", y="PC1"):
    _, tutorial_index = get_tutorial_df(subject_data)
    _, prescan_index = get_prescan_df(subject_data, prev_index=tutorial_index)
    x_coords = [subject_data[x].loc[tutorial_index],
         subject_data[x].loc[prescan_index],
         subject_data[x].iloc[-1]]
    y_coords = [subject_data[y].loc[tutorial_index],
                subject_data[y].loc[prescan_index],
                subject_data[y].iloc[-1]]
    return x_coords, y_coords

def get_time_series_slopes(subject_dfs, index_ordered_list, time_col="DurationElapsed"):
    """Function for returning the slopes of each phase's time series"""
    slope_series = pd.Series(index=index_ordered_list)
    for slope_name, subject_df in zip(index_ordered_list, subject_dfs):
        if "All" or "Tutorial" in slope_name:
            subject_lm = LinearRegression(fit_intercept=False)
        else:
            subject_lm = LinearRegression(fit_intercept=True)
        subject_lm.fit(X=np.array(subject_df.loc[:, time_col].values.reshape(-1, 1)) / 60,
                           y=np.array(subject_df.loc[:, "PC1"]))
        slope_series[slope_name] = subject_lm.coef_[0]
    return slope_series


def get_gold_series_distances(df_pairs, index_ordered_list):
    """Function for returning the distances to the gold path of each student across each phase"""
    distance_series = pd.Series(index=index_ordered_list)
    for distance_name, df_pair in zip(index_ordered_list, df_pairs):
        reference_df, subject_data = df_pair[0], df_pair[1]
        subj_avg, subj_std, subj_max = tci.series_distance(seq_one=reference_df.loc[:, ["PC1", "TimeStamp"]],
                                                            seq_two=subject_data,
                                                            distance_features=["PC1"],
                                                            length_mismatch="padded")
        distance_series[distance_name] = subj_avg
    return distance_series


def get_time_series_subject_phase_break_df(event_df, gold_df, omit_list):
    """Given the event dataframe, return a dictionary mapping test subjects to """
    subjects = [e for e in list(event_df.loc[:, "TestSubject"].unique()) if e not in omit_list and pd.notna(e)]

    slope_ordered_list = ["Slope-All", "Slope-Tutorial", "Slope-PreScan", "Slope-PostScan"]
    dist_ordered_list = ["Dist-All", "Dist-Tutorial", "Dist-PreScan", "Dist-PostScan"]


    # Get phase event dataframes for gold path
    gold_tutorial, gold_tutorial_end_index = get_tutorial_df(gold_df)  # Make sure Elise Beach annotated in Gold Path
    gold_prescan, gold_prescan_end_index = get_prescan_df(gold_df, prev_index=gold_tutorial_end_index)
    gold_postscan = get_postscan_df(gold_df, prev_index=gold_prescan_end_index)
    gold_slopes = get_time_series_slopes([gold_df, gold_tutorial, gold_prescan, gold_postscan],
                                         time_col="GameTime", index_ordered_list=slope_ordered_list)

    return_data_df = pd.DataFrame(pd.concat([gold_slopes, pd.Series(np.zeros(4))]).values.reshape(1, -1),
                                  index=["GOLD"], columns=slope_ordered_list + dist_ordered_list)
    for subject in subjects:
        # Get subject rows
        subject_rows = event_df["TestSubject"] == subject
        subject_df = event_df.loc[subject_rows, :]
        subject_df.index = list(range(subject_df.shape[0]))

        subject_start_time = parse(subject_df["TimeStamp"].iloc[0])
        subject_df["DurationElapsed"] = subject_df.loc[:, "TimeStamp"].apply(
            lambda cell: (parse(cell) - subject_start_time).total_seconds())

        # Get each phase event dataframe for subject
        subject_tutorial, tutorial_end_index = get_tutorial_df(subject_df)
        subject_prescan, prescan_end_index = get_prescan_df(subject_df, prev_index=tutorial_end_index)
        subject_postscan = get_postscan_df(subject_df, prev_index=prescan_end_index)

        try:
            slope_series = get_time_series_slopes([subject_df, subject_tutorial, subject_prescan, subject_postscan],
                                              time_col="DurationElapsed", index_ordered_list=slope_ordered_list)
        except ValueError:
            print(subject)

        gold_distance_series = get_gold_series_distances([(gold_df, subject_df),
                                                          (gold_tutorial, subject_tutorial),
                                                          (gold_prescan, subject_prescan),
                                                          (gold_postscan, subject_postscan)],
                                                         index_ordered_list=dist_ordered_list)

        subject_result_df = pd.DataFrame(pd.concat([slope_series, gold_distance_series]).values.reshape(1, -1),
                                         index=[subject], columns=slope_ordered_list + dist_ordered_list)
        return_data_df = pd.concat([return_data_df, subject_result_df], axis=0)
    return return_data_df

if __name__ == "__main__":
    directory = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/"
    golden_df_intermediate_file = directory + "Intermediate/GoldenDF.csv"
    full_df_intermediate_file = directory + "Intermediate/FullDF.csv"
    output_filename = directory + "Intermediate/MetricsOutput.csv"

    ig_df = pd.read_csv(golden_df_intermediate_file)
    full_df = pd.read_csv(full_df_intermediate_file)

    slope_dist_interval_data = get_time_series_subject_phase_break_df(event_df=full_df,
                                                                     gold_df=ig_df,
                                                                     omit_list=[])
    slope_dist_interval_data.to_csv(output_filename, index=True)
