# Script for running other scripts (master script) to create data from scratch (from data pipeline output)

from facet_event_file import create_facet_event_file
from merge_event_files import merge_event_files
from separate_event_into_intervals import parse_event_into_intervals
from windowed_actions import create_windowed_action_file
from windowed_action_analysis import perform_condition_analysis
from windowed_action_analysis import perform_action_analysis
from grade_qualtrics import grade_tests
from add_interval_durations import add_interval_durations
from add_action_counts import add_action_counts_by_interval
from add_game_score import add_game_score
from add_action_counts import add_cumulative_action_counts
import pandas as pd
from reformat_output import parse_script
from facet_diagnostics import output_FACET_evidence


def remove_aois(event_filename, output_filename):
    df = pd.read_csv(event_filename)
    keep_rows = df["Event"] != "AOI"
    df = df.loc[keep_rows,:]
    df.to_csv(output_filename, index=False)


if __name__ == "__main__":
    # Local system filepaths, raw = from pipeline, desired = created by scripts (intermediate or final), retrieved = not from pipeline but output from somewhere (possible manual or download)

    # Change prefix based on location of Data Pipeline Output folder
    ci_prefix = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output318/"

    # Retrieved files are not part of the data pipeline and do not have the prefix as they could be stored in alternate places
    retrieved_qualtrics_key = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/SurveyGradingScheme.csv"
    retrieved_post_test = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/CI_PostTest_Data 3-1-18.csv"
    retrieved_pre_test = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/CI_PreTest_Data 3-1-18.csv"

    # These should not need to be changed (output from data pipeline in the ci_prefix directory)
    unfiltered_event_sequence = ci_prefix + "EventSequence/EventSequence.csv"
    raw_event_sequence = ci_prefix + "EventSequence/EventSequenceP.csv"
    raw_facet_threshold = ci_prefix + "FACET-ThresholdCrossed/FACET-ThresholdCrossed.csv"
    raw_activity_summary = ci_prefix + "ActivitySummary/ActivitySummary.csv"
    raw_books_articles = ci_prefix + "BooksAndArticles/BooksAndArticles.csv"
    raw_facet_threshold_directory = ci_prefix + "FACET-Z-Score/"
    raw_facet_directory = ci_prefix + "FACET/"

    # Change as needed based on desired filenames (will be output to ci_prefix directory)
    desired_graded_activity_summary = ci_prefix + "ActivitySummary/ActivitySummaryGraded.csv"
    desired_facet_events = ci_prefix + "FACET-ThresholdCrossed/FACET-Events.csv"
    desired_event_facet = ci_prefix + "EventSequence/EventSequenceFACET.csv"
    desired_event_facet_no_aoi = ci_prefix + "EventSequence/EventSequenceFACETNoAOI.csv"
    desired_event_no_facet = ci_prefix + "EventSequence/EventSequence_wCGS_NoAOI.csv"
    desired_windowed_prefix = ci_prefix + "EventSequence/WindowedActions"
    desired_interval_activity_summary = ci_prefix + "ActivitySummary/ActivitySummaryInterval.csv"
    desired_appended_activity_summary = ci_prefix + "ActivitySummary/ActivitySummaryAppended.csv"
    desired_standardized_activity_summary = ci_prefix + "ActivitySummary/ActivitySummaryStd.csv"

    # Arguments for functions
    window_size = 5  # number of seconds for the before/after action to window facet emotions
    duration_min = 0.0  # number of seconds to use as minimum time to qualify as an emotion event (default = 0.5)

    # Removing line break characters that cause bugs later
    parse_script(unfiltered_event_sequence, raw_event_sequence)

    # Function calls in proper order for generation of intermediate files
    create_facet_event_file(facet_thresh_filename=raw_facet_threshold,
                            facet_event_filename=desired_facet_events,
                            event_filename=raw_event_sequence,
                            only_positives=True,
                            duration_min=duration_min)
    merge_event_files(event_filename=raw_event_sequence,
                      facet_filename=desired_facet_events,
                      event_facet_filename=desired_event_facet,
                      show=False)

    grade_tests(grading_key_filename=retrieved_qualtrics_key,
                post_test_filename=retrieved_post_test,
                pre_test_filename=retrieved_pre_test,
                activity_summary_filename=raw_activity_summary,
                output_filename=desired_graded_activity_summary,
                normalize=True)

    add_game_score(event_filename=desired_event_facet,
                   activity_summary=desired_graded_activity_summary,
                   books_articles=raw_books_articles,
                   appended_event_filename=desired_event_facet,
                   appended_activity_summary=desired_graded_activity_summary)

    add_cumulative_action_counts(event_sequence_filename=desired_event_facet,
                                 event_sequence_cumulatives_output=desired_event_facet,
                                 desired_event_names=None,
                                 desired_locations=None,
                                 perform_parse_locations=True,
                                 remove_aois=False,
                                 include_facet=True,
                                 show=False)

    parse_event_into_intervals(event_facet_filename=desired_event_facet)  # auto-generates output filenames using nomenclature from this file

    perform_condition_analysis(data_filename=desired_graded_activity_summary,
                               output_filename=desired_graded_activity_summary[:-4]+"-ConditionAnalysis.csv")

    add_interval_durations(event_sequence_filename=raw_event_sequence,
                           activity_summary_filename=desired_graded_activity_summary,
                           output_filename=desired_interval_activity_summary)

    add_action_counts_by_interval(activity_summary_filename=desired_interval_activity_summary,
                                  event_sequence_filename=desired_event_facet,
                                  output_filename=desired_appended_activity_summary,
                                  std_output_filename=desired_standardized_activity_summary,
                                  intervals=["", "Tutorial", "PreScan", "PostScan"])

    perform_condition_analysis(data_filename=desired_standardized_activity_summary,
                               output_filename=desired_standardized_activity_summary[:-4]+"-ConditionAnalysis.csv")

    # create_windowed_action_file(event_facet_filename=desired_event_facet[:-4]+"Tutorial.csv",
    #                             output_filename=desired_windowed_prefix+"Tutorial.csv", window_size=window_size)
    # create_windowed_action_file(event_facet_filename=desired_event_facet[:-4] + "PreScan.csv",
    #                             output_filename=desired_windowed_prefix + "PreScan.csv", window_size=window_size)
    # create_windowed_action_file(event_facet_filename=desired_event_facet[:-4] + "PostScan.csv",
    #                             output_filename=desired_windowed_prefix + "PostScan.csv", window_size=window_size)
    # create_windowed_action_file(event_facet_filename=desired_event_facet,
    #                             output_filename=desired_windowed_prefix + "All.csv", window_size=window_size)

    perform_condition_analysis(data_filename=desired_windowed_prefix+"Tutorial.csv",
                               output_filename=desired_windowed_prefix+"Tutorial-ConditionAnalysis.csv")
    perform_condition_analysis(data_filename=desired_windowed_prefix+"PreScan.csv",
                               output_filename=desired_windowed_prefix+"PreScan-ConditionAnalysis.csv")
    perform_condition_analysis(data_filename=desired_windowed_prefix+"PostScan.csv",
                               output_filename=desired_windowed_prefix+"PostScan-ConditionAnalysis.csv")
    perform_condition_analysis(data_filename=desired_windowed_prefix+"All.csv",
                               output_filename=desired_windowed_prefix+"All-ConditionAnalysis.csv")

    perform_action_analysis(windowed_action_filename=desired_windowed_prefix+"Tutorial.csv",
                               output_filename=desired_windowed_prefix+"Tutorial-ActionAnalysis.csv")
    perform_action_analysis(windowed_action_filename=desired_windowed_prefix+"PreScan.csv",
                               output_filename=desired_windowed_prefix+"PreScan-ActionAnalysis.csv")
    perform_action_analysis(windowed_action_filename=desired_windowed_prefix+"PostScan.csv",
                               output_filename=desired_windowed_prefix+"PostScan-ActionAnalysis.csv")
    perform_action_analysis(windowed_action_filename=desired_windowed_prefix+"All.csv",
                               output_filename=desired_windowed_prefix+"All-ActionAnalysis.csv")

    # student_df = output_FACET_evidence(target_dir=raw_facet_threshold_directory,
    #                                    data_extension="FACET-Z-Score.csv",
    #                                    facet_event_filename=desired_facet_events,
    #                                    output_diagnostics=True,
    #                                    output_correlations=True,
    #                                    output_figures=True,
    #                                    emotions_to_use=["Confusion", "Joy", "Frustration"],
    #                                    include_aus=True,
    #                                    merged=True)

    # student_df = output_FACET_evidence(target_dir=raw_facet_directory,
    #                                    data_extension="FACET.csv",
    #                                    facet_event_filename=desired_facet_events,
    #                                    output_diagnostics=True,
    #                                    output_correlations=True,
    #                                    output_figures=True,
    #                                    emotions_to_use=["Confusion", "Joy", "Frustration"],
    #                                    include_aus=True,
    #                                    merged=True)








