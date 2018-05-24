import pandas as pd
import numpy as np
import csv
import datetime


ACTION_TYPES = {"KnowledgeAcquisition": ["BooksAndArticles", "PostersLookedAt"],
                "InformationGathering": ["Conversation"],
                "HypothesisTesting": ["Worksheet", "WorksheetSubmit", "Scanner"]}


def unstack_df(df, prefix, suffix="Count"):
    df = df.unstack(level=1)
    df.columns = ["%s-%s-%s" % (prefix, e, suffix) for e in df.columns]
    df["TestSubject"] = df.index
    df.index = range(df.shape[0])
    return df


def append_cumulative_action_counts(event_sequence_filename, activity_summary_filename, appended_activity_summary_filename):
    """Function for adding the final action count row from EventSequenceCumulatives.csv to the activity summary"""
    event_sequence = pd.read_csv(event_sequence_filename)
    cumulative_cols = [e for e in event_sequence.columns if "C-" in e]
    act_sum = pd.read_csv(activity_summary_filename)
    act_sum.index = act_sum["TestSubject"]
    for col in cumulative_cols:
        act_sum[col] = pd.Series([0] * act_sum.shape[0])
    subjects = list(act_sum["TestSubject"].unique())

    for subject in subjects:
        subject_event_rows = event_sequence["TestSubject"] == subject
        if sum(subject_event_rows) >= 1:
            subject_events = (event_sequence.loc[subject_event_rows, :]).iloc[-1, :]
            subject_counts = subject_events.loc[cumulative_cols]
            act_sum.loc[subject, cumulative_cols] = subject_counts
    act_sum.to_csv(appended_activity_summary_filename, index=False)


def add_action_counts_by_interval(activity_summary_filename, event_sequence_filename, output_filename, std_output_filename, intervals):
    """
    Use the interval specific event files to determine count and duration of actions and emotions by interval
        These are then appended to the activity summary
    :param activity_summary_filename: String - filename of the csv containing the action summary
    :param event_sequence_filename: String - base filename for the interval specific event sequences (with previously merged in FACET events)
    :param output_filename: String - filename to output the appended activity summary
    :param std_output_filename: String - filename to output the appended and standardized by m=0 s=1 activity summary
    :param intervals: List of Strings - extensions for the interval specific filenames
    :return: None, outputs to specific filenames
    """
    act_sum_df = pd.read_csv(activity_summary_filename)
    all_columns = []

    for interval in intervals:
        event_df = pd.read_csv(event_sequence_filename[:-4]+interval+".csv", quoting=csv.QUOTE_NONE)
        event_df.fillna(0, inplace=True)

        if interval == "":
            prefix = "All"
        else:
            prefix = interval
        count_df = event_df.groupby("TestSubject")["Event"].value_counts()
        count_df = unstack_df(count_df, prefix=prefix, suffix="Count")
        duration_df = event_df.groupby(["TestSubject", "Event"])["Duration"].sum()
        duration_df = unstack_df(duration_df, prefix=prefix, suffix="Duration")
        try:
            duration_df["%s-WorksheetSubmit-Duration" % prefix] = count_df["%s-WorksheetSubmit-Count" % prefix]
        except KeyError:
            pass  # Indicates no worksheet submits in the interval
        emotion_rows = event_df["Event"] == "FACET"
        emotion_df = event_df.loc[emotion_rows, :]
        emotion_count_df = emotion_df.groupby("TestSubject")["Target"].value_counts()
        emotion_count_df = unstack_df(emotion_count_df, prefix=prefix, suffix="Count")
        emotion_duration_df = emotion_df.groupby(["TestSubject", "Target"])["Duration"].sum()
        emotion_duration_df = unstack_df(emotion_duration_df, prefix=prefix, suffix="Duration")

        additional_columns = []
        act_sum_df = act_sum_df.merge(right=count_df, how='left', on="TestSubject")
        additional_columns += list(count_df.columns.values)
        all_columns += list(count_df.columns.values)
        act_sum_df = act_sum_df.merge(right=duration_df, how='left', on="TestSubject")
        additional_columns += list(duration_df.columns.values)
        all_columns += list(duration_df.columns.values)
        act_sum_df = act_sum_df.merge(right=emotion_count_df, how='left', on="TestSubject")
        additional_columns += list(emotion_count_df.columns.values)
        all_columns += list(emotion_count_df.columns.values)
        act_sum_df = act_sum_df.merge(right=emotion_duration_df, how='left', on="TestSubject")
        additional_columns += list(emotion_duration_df.columns.values)
        all_columns += list(emotion_duration_df.columns.values)

        if interval == "":
            prefix = ""
        else:
            prefix = "-"
        # For each additional column, divide by the duration spent in interval (in minutes)
        # This means numbers reported in activity summary are rate/proportion for count/duration respectively
        for add_col in additional_columns:
            try:
                act_sum_df[add_col] = act_sum_df[add_col].divide(act_sum_df["Duration%s%s" % (prefix, interval)]/60., fill_value=0.1)
            except TypeError:
                if add_col != "TestSubject":
                    print("Error with %s" % add_col)

        print("Finished %s interval" % interval)

    act_sum_df = act_sum_df.replace(np.inf, 0)
    std_act_sum = act_sum_df.copy()
    for add_col in all_columns:
        if add_col != "TestSubject":
            std_act_sum[add_col] = (std_act_sum[add_col] - std_act_sum[add_col].mean()) / std_act_sum[add_col].std()
    for interval in ["All", "Tutorial", "PreScan", "PostScan"]:
        for key in ACTION_TYPES.keys():
            std_act_sum["%s-%s-Count" % (interval, key)] = np.zeros(std_act_sum.shape[0])
            std_act_sum["%s-%s-Duration" % (interval, key)] = np.zeros(std_act_sum.shape[0])
            for value in ACTION_TYPES[key]:
                try:
                    std_act_sum["%s-%s-Count" % (interval, key)] += std_act_sum["%s-%s-Count" % (interval, value)]
                    std_act_sum["%s-%s-Duration" % (interval, key)] += std_act_sum["%s-%s-Duration" % (interval, value)]
                except KeyError:
                    pass  # Likely means that the interval does not contain that action
    std_act_sum.to_csv(std_output_filename, index=False)
    act_sum_df.to_csv(output_filename, index=False)


def _get_stem(loc):
    """For specific CamelCase word, return the first word"""
    try:
        index = -1
        for char, i in zip(loc, range(len(loc))):
            if char.isupper() and i != 0 and index == -1: # Uppercase letter that is not the first letter (all caps) and is not reset by later caps
                index = i
        if index == -1:
            return loc
        else:
            return loc[:index]
    except TypeError:
        return "INVALID"


def parse_locations(location_list):
    """Return all unique 'stems' of the locations, as first word of location is typically a higher level (larger) location"""
    parsed_locations = []
    for loc in location_list:
        stem = _get_stem(loc)
        if stem not in parsed_locations and stem != "INVALID":
            parsed_locations.append(stem)
    return parsed_locations


def add_cumulative_action_counts(event_sequence_filename, event_sequence_cumulatives_output, desired_event_names=None, desired_locations=None, desired_facets=None, perform_parse_locations=True, remove_aois=True, include_facet=True, show=False):
    """Function for adding several columns to EventSequence.csv (or EventSequenceFACET.csv) 
    for cumulative counts of actions and locations visited"""
    start_time = datetime.datetime.now()
    print("Started adding cumulative action counts: %s" % start_time)
    event_data = pd.read_csv(event_sequence_filename)
    event_names = list(event_data.loc[:, "Event"].unique())
    location_names = [e for e in list(event_data.loc[:, "Location"].unique()) if pd.notnull(e)]
    facet_rows = event_data["Event"] == "FACET"
    facet_names = list(event_data.loc[facet_rows, "Target"].unique())

    if perform_parse_locations:
        location_names = parse_locations(location_names)

    event_counts = dict(zip(event_names, np.zeros(len(event_names))))
    event_durs = dict(zip(event_names, np.zeros(len(event_names))))
    location_counts = dict(zip(location_names, np.zeros(len(location_names))))
    facet_counts = dict(zip(facet_names, np.zeros(len(facet_names))))
    facet_durations = dict(zip(facet_names, np.zeros(len(facet_names))))

    if desired_event_names is None:
        desired_event_names = event_names
    if desired_locations is None:
        desired_locations = location_names
    if desired_facets is None:
        desired_facets = facet_names

    new_rows = []

    current_id = 'none'
    prev_location = 'none'
    for index, row in event_data.iterrows():
        if current_id != row["TestSubject"]:
            if show:
                print("%s | %s | %s " % (current_id, event_counts, location_counts))
            prev_location = 'none'
            current_id = row["TestSubject"]
            event_counts = dict(zip(event_names, np.zeros(len(event_names))))
            event_durs = dict(zip(event_names, np.zeros(len(event_names))))
            location_counts = dict(zip(location_names, np.zeros(len(location_names))))
            facet_counts = dict(zip(facet_names, np.zeros(len(facet_names))))
            facet_durations = dict(zip(facet_names, np.zeros(len(facet_names))))

        if row["Event"] == "FACET":
            event_counts["FACET"] += 1
            facet_counts[row["Target"]] += 1
            facet_durations[row["Target"]] += float(row["Duration"]) if pd.notnull(float(row["Duration"])) else 0
        else:
            event_counts[row["Event"]] += 1
            event_durs[row["Event"]] += float(row["Duration"]) if pd.notnull(float(row["Duration"])) else 0
        current_location = _get_stem(row["Location"])
        if prev_location != current_location and current_location != "INVALID":
            location_counts[current_location] += 1

        prev_location = current_location
        current_row = [event_counts[key] for key in desired_event_names] \
                      + [location_counts[key] for key in desired_locations] \
                      + [facet_counts[key] for key in desired_facets] \
                      + [facet_durations[key] for key in desired_facets] \
                      + [event_durs[key] for key in desired_event_names]
        new_rows.append(current_row)

    new_cols = ["C-A-%s" % event_name for event_name in desired_event_names] \
               + ["C-L-%s" % loc_name for loc_name in desired_locations] \
               + ["C-F-%s" % fac_name for fac_name in desired_facets] \
               + ["C-FD-%s" % fac_name for fac_name in desired_facets] \
               + ["C-D-%s" % event_name for event_name in desired_event_names]
    new_data = pd.DataFrame(new_rows, columns=new_cols)
    full_data = pd.concat([event_data, new_data], axis=1)
    if remove_aois:
        non_aoi_rows = full_data.loc[:, "Event"] != "AOI"
        full_data = full_data.loc[non_aoi_rows, :]
        if "C-A-AOI" in full_data.columns:
            full_data.drop(labels="C-A-AOI", axis=1, inplace=True)
    full_data.to_csv(event_sequence_cumulatives_output, index=False, doublequote=False)
    print("Finished adding cumulative action counts in %.4f minutes" % ((datetime.datetime.now() - start_time).total_seconds()/60.0))

if __name__ == "__main__":
    event_sequence_filename = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/EventSequence/EventSequenceFACET.csv"
    activity_summary_filename = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/ActivitySummary/ActivitySummaryGradedIntervals.csv"
    output_filename = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/ActivitySummary/ActivitySummaryGradedActions.csv"
    std_output_filename = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/ActivitySummary/ActivitySummaryStd.csv"
    intervals = ["", "Tutorial", "PreScan", "PostScan"]

    # desired_event_names = ["Conversation", "BooksAndArticles", "Worksheet", "WorksheetSubmit", "Scanner"]
    # add_action_counts_by_interval(activity_summary_filename=activity_summary_filename,
    #                               event_sequence_filename=event_sequence_filename,
    #                               output_filename=output_filename,
    #                               intervals=intervals,
    #                               std_output_filename=std_output_filename)
    # dir_stem = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/OutputFull2018/"
    # event_sequence_filename = dir_stem + "EventSequence/EventSequenceFACET-All.csv"
    # activity_summary_filename = dir_stem + "ActivitySummary/ActivitySummaryInterval.csv"
    # output_filename = dir_stem + "ActivitySummary/ActivitySummaryAppended-All.csv"
    # std_output_filename = dir_stem + "ActivitySummary/ActivitySummaryStd-All.csv"
    # add_action_counts_by_interval(activity_summary_filename=activity_summary_filename,
    #                               event_sequence_filename=event_sequence_filename,
    #                               output_filename=output_filename,
    #                               intervals=intervals,
    #                               std_output_filename=std_output_filename)

    event_seq = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/OutputFull2018/EventSequence/EventSequenceP.csv"
    event_seq_c = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/OutputFull2018/EventSequence/EventSequenceCumulatives.csv"
    act_sum = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/OutputFull2018/ActivitySummary/ActivitySummaryAppendedRevisedEdited-All.csv"
    act_sum_appended = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/OutputFull2018/ActivitySummary/ActivitySummaryTransfer.csv"
    # add_cumulative_action_counts(event_sequence_filename=event_seq,
    #                              event_sequence_cumulatives_output=event_seq_c)
    append_cumulative_action_counts(event_sequence_filename=event_seq_c,
                                    activity_summary_filename=act_sum,
                                    appended_activity_summary_filename=act_sum_appended)

