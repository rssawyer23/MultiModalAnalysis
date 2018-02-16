# Script for getting FACET scores over key intervals of MetaTutorIVH data
#   Specifically, for getting Pre-Tutor, During-Tutor, and Post-Tutor average FACET scores

import pandas as pd
import numpy as np
from dateutil.parser import parse
import datetime


class TrialData:
    """Object for keeping times and facet data for intervals of interest during a trial"""
    def __init__(self, trial_id, facet_keys, include_negatives):
        self.id = trial_id
        self.desired_facets = list(facet_keys)
        self.include_negatives = include_negatives
        self.time_dict = dict()
        self.pre_facet_lists = dict()  # Mapping from FACET type to list of measurements for pre-tutor phase
        self.during_facet_lists = dict()
        self.post_facet_lists = dict()

        for facet_type in self.desired_facets:
            self.pre_facet_lists[facet_type] = []
            self.during_facet_lists[facet_type] = []
            self.post_facet_lists[facet_type] = []

    def initialize_facet_lists(self, desired_facets):
        self.desired_facets = list(desired_facets)
        for facet_type in desired_facets:
            self.pre_facet_lists[facet_type] = []
            self.during_facet_lists[facet_type] = []
            self.post_facet_lists[facet_type] = []

    def determine_phase(self, parsed_timestamp):
        if parsed_timestamp < self.time_dict["Content Start"]:
            phase_type = "INVALID"
        elif parsed_timestamp < self.time_dict["Tutor Start"]:
            phase_type = "Pre"
        elif parsed_timestamp < self.time_dict["Tutor End"]:
            phase_type = "During"
        elif parsed_timestamp < self.time_dict["Content End"]:
            phase_type = "Post"
        else:
            phase_type = "INVALID"
        return phase_type

    def add_facet_row(self, facet_row):
        parsed_timestamp = parse(facet_row["TimeStamp"])
        phase_key = self.determine_phase(parsed_timestamp)
        if phase_key == "Pre":
            for e in self.desired_facets:
                try:
                    value = float(facet_row[e + "Evidence"])
                    if not self.include_negatives and value < 0:
                        value = 0
                    self.pre_facet_lists[e].append(value)
                except ValueError:
                    pass
        elif phase_key == "During":
            for e in self.desired_facets:
                try:
                    value = float(facet_row[e + "Evidence"])
                    if not self.include_negatives and value < 0:
                        value = 0
                    self.during_facet_lists[e].append(value)
                except ValueError:
                    pass
        elif phase_key == "Post":
            for e in self.desired_facets:
                try:
                    value = float(facet_row[e + "Evidence"])
                    if not self.include_negatives and value < 0:
                        value = 0
                    self.post_facet_lists[e].append(value)
                except ValueError:
                    pass
        elif phase_key == "INVALID":
            pass
        else:
            print("Unrecognized key %s" % phase_key)

    def output_facet_means(self, ordered_facets):
        output_series = pd.Series()
        for f in ordered_facets:
            try:
                output_series["Pre-%s" % f] = np.mean(self.pre_facet_lists[f])
            except KeyError:
                output_series["Pre-%s" % f] = 0.0
            try:
                output_series["During-%s" % f] = np.mean(self.during_facet_lists[f])
            except KeyError:
                output_series["During-%s" % f] = 0.0
            try:
                output_series["Post-%s" % f] = np.mean(self.post_facet_lists[f])
            except KeyError:
                output_series["Post-%s" % f] = 0.0
        return output_series


class TestSubject:
    def __init__(self, subject_id):
        self.id = subject_id
        self.trial_data = []
        self.trial_counter = 0

    def add_trial_data(self, trial_data):
        """Add a TrialData object to list of trials here"""
        self.trial_data.append(trial_data)

    def print_trials(self):
        """Output the subject id and list of trials associated with subejct to ensure correct data"""
        output_list = []
        for t in self.trial_data:
            output_list.append(t.id)
        print("%s: %s" % (self.id, output_list))

    def allocate_data(self, facet_row, desired_facets):
        parsed_timestamp = parse(facet_row["TimeStamp"])
        while self.trial_counter < len(self.trial_data) and self.trial_data[self.trial_counter].time_dict["Content End"] < parsed_timestamp:
            self.trial_counter += 1
        if self.trial_counter < len(self.trial_data):
            if len(self.trial_data[self.trial_counter].desired_facets) == 0:
                self.trial_data[self.trial_counter].initialize_facet_lists(desired_facets)
            self.trial_data[self.trial_counter].add_facet_row(facet_row)
        else:
            pass  # Beyond final trial content page, ignore

    def get_trial_data_frame(self, ordered_facets):
        return_df = pd.DataFrame()
        for t in self.trial_data:
            trial_series = t.output_facet_means(ordered_facets)
            return_df = pd.concat([return_df, trial_series], axis=1)

        return_df = return_df.T
        return_df.index = list(range(return_df.shape[0]))
        return_df["TrialId"] = pd.Series([t.id for t in self.trial_data])
        return return_df


def create_trial_timing_dictionary(es_df, trial_expressions, desired_facets, include_negatives):
    """Using the EventSequence and Trials file, create a dictionary mapping each TrialId to a dictionary containing key times for intervals"""
    trials = list(es_df["TrialId"].unique())
    test_subject_dict = dict()

    for trial in trials:
        trial_rows = es_df["TrialId"] == trial
        trial_df = es_df.loc[trial_rows, :]
        trial_obj = TrialData(trial, desired_facets, include_negatives=include_negatives)

        test_subject = trial_df["TestSubject"].iloc[0]
        if test_subject not in test_subject_dict.keys():
            test_subject_obj = TestSubject(test_subject)
            test_subject_dict[test_subject] = test_subject_obj

        start_time_row = (trial_df["Event"] == "Pages") & (trial_df["PageName"] == "ContentPage")
        start_time = trial_df.loc[start_time_row, "TimeStamp"].iloc[0]
        tutor_time_row = (trial_df["Event"] == "SubmittedResponses") & (es_df["Field"] == "DiagramJudgmentResponse")
        tutor_time = trial_df.loc[tutor_time_row, "TimeStamp"].iloc[0]
        end_time_row = (trial_df["Event"] == "Pages") & (trial_df["PageName"] == "MultipleChoiceQuestionPage")
        end_time = trial_df.loc[end_time_row, "TimeStamp"].iloc[0]

        expression_seconds = 2.2 if trial_expressions.loc[trial, "AgentReaction"] == "Neutral" else 12.0

        trial_obj.time_dict = {"Content Start": parse(start_time),
                             "Tutor Start": parse(tutor_time),
                             "Tutor End": parse(tutor_time) + datetime.timedelta(seconds=expression_seconds),
                             "Content End": parse(end_time)}

        if trial_obj.time_dict["Tutor End"] > trial_obj.time_dict["Content End"]:
            print("ERROR WITH TRIAL %s TIMES NOT IN PROPER ORDER" % trial)
        else:
            test_subject_dict[test_subject].add_trial_data(trial_obj)
    return test_subject_dict


def index_dictionary(header_line):
    in_dict = dict()
    header_split = header_line.replace("\n","").replace("\r","").split(",")
    for ele, index in zip(header_split,range(len(header_split))):
        in_dict[ele] = index
    return in_dict


def _create_dict_row(kim, line, desired_facets):
    """Alternative to using each row as series under pandas framework"""
    row_dict = dict()
    split = line.replace("\n", "").split(",")
    row_dict["TestSubject"] = split[kim["TestSubject"]]
    row_dict["TimeStamp"] = split[kim["TimeStamp"]]
    for e in desired_facets:
        row_dict[e+"Evidence"] = split[kim[e+"Evidence"]]
    return row_dict


def create_merged_trial_facet_file(event_sequence_filepath, trials_filepath, facet_filepath, trial_facet_filepath, desired_facets, include_negatives, show):
    start_time = datetime.datetime.now()
    es_df = pd.read_csv(event_sequence_filepath)
    page_rows = (es_df["Event"] == "Pages") & ((es_df["PageName"] == "ContentPage") | (es_df["PageName"] == "MultipleChoiceQuestionPage"))
    submit_rows = (es_df["Event"] == "SubmittedResponses") & (es_df["Field"] == "DiagramJudgmentResponse")
    necessary_rows = np.logical_or(page_rows, submit_rows)
    es_df = es_df.loc[necessary_rows, :]

    trial_df = pd.read_csv(trials_filepath)
    trial_expressions = trial_df.loc[:, ["TrialId", "AgentReaction"]]
    trial_expressions.index = trial_expressions.loc[:, "TrialId"]
    trial_expressions.drop(labels="TrialId", axis=1, inplace=True)

    if show:
        print("Started creating subject trials data: %s" % datetime.datetime.now())
    subject_trials = create_trial_timing_dictionary(es_df=es_df,
                                                    trial_expressions=trial_expressions,
                                                    desired_facets=desired_facets,
                                                    include_negatives=include_negatives)

    if show:
        print("Started Loading FACET file: %s" % datetime.datetime.now())
    facet_data_file = open(facet_filepath, 'r')
    header = facet_data_file.readline()
    kim = index_dictionary(header)  # Key Index Map

    if show:
        print("Started Reading FACET file: %s" % datetime.datetime.now())

    lines_processed = 0
    index_errors = 0
    for line in facet_data_file:
        try:
            if lines_processed % 4 == 0:
                row = _create_dict_row(kim, line, desired_facets)
                subject_trials[row["TestSubject"]].allocate_data(row, desired_facets)
        except IndexError:
            index_errors += 1
        lines_processed += 1
        if lines_processed % 50000 == 0 and show:
            print("Processed %d lines, approximately %.4f percent %s" % (lines_processed, lines_processed/5770000 * 100, datetime.datetime.now()))

    if show:
        print("Finished reading FACET file with %d errors: %s" % (index_errors, datetime.datetime.now()))
        print("Started gathering FACET data into output format %s" % datetime.datetime.now())

    full_df = pd.DataFrame()
    for key in subject_trials.keys():
        subject_df = subject_trials[key].get_trial_data_frame(desired_facets)
        full_df = pd.concat([full_df, subject_df])
    merged_df = trial_df.merge(right=full_df, how='left', on="TrialId")
    merged_df.to_csv(trial_facet_filepath, index=False)
    if show:
        print("Finished in %.2f minutes" % ((datetime.datetime.now() - start_time).total_seconds()/60.0))
    return merged_df


if __name__ == "__main__":
    # Should only need to change metatutor_directory to follow your local filepath to TraceDataPipeline Output folder
    metatutor_directory = "C:/Users/robsc/Documents/NC State/GRAWork/IVHData/OutputJan18Neg/"

    event_sequence_fp = metatutor_directory + "EventSequence/EventSequence.csv"
    trials_fp = metatutor_directory + "Trials/Trials.csv"
    facet_fp = metatutor_directory + "FACET/FACETP.csv"   # Likely do not have this file, run reformat_output.py to generate
    trial_facet_fp = metatutor_directory + "Trials/Trials_wFACET_0.csv"  # THIS IS THE OUTPUT FILE, NAME AS DESIRED

    desired_emotions = ["Joy", "Confusion", "Frustration"]
    include_negatives = False  # Setting this to False will zero out any negative FACET scores when calculating averages
    show = True  # If True the console will output some timing information (overall should take less than 10 minutes)

    # Will output new Trials.csv into
    result_df = create_merged_trial_facet_file(event_sequence_filepath=event_sequence_fp,
                                               trials_filepath=trials_fp,
                                               facet_filepath=facet_fp,
                                               trial_facet_filepath=trial_facet_fp,
                                               desired_facets=desired_emotions,
                                               include_negatives=include_negatives,
                                               show=show)
