import pandas as pd
import windowed_actions
import add_action_counts
import reformat_output
import facet_event_file
import merge_event_files

if __name__ == "__main__":
    omit_AOIs = True  # if True only the 9 emotions are included, if false, all emotions + AUs included
    omit_AUs = True
    show = True
    window_size = 5  # in seconds, for the before/after action window size
    directory = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output318/"
    unfiltered_event_sequence = directory + "EventSequence/EventSequence.csv"
    raw_event_sequence = directory + "EventSequence/EventSequenceP.csv"
    raw_facet_threshold = directory + "FACET-ThresholdCrossed/FACET-ThresholdCrossed.csv"
    raw_activity_summary = directory + "ActivitySummary/ActivitySummary.csv"
    raw_books_and_articles = directory + "BooksAndArticles/BooksAndArticles.csv"

    desired_facet_events = directory + "FACET-ThresholdCrossed/FACET-Events.csv"
    desired_event_facet = directory + "EventSequence/EventSequenceFACET.csv"
    desired_event_cumulatives_filename = directory + "EventSequence/EventSequenceCumulatives.csv"
    desired_windowed_action_filename = directory + "EventSequence/WindowedActionsAll.csv"

    out_directory = "C:/Users/robsc/Documents/NC State/GRAWork/Publications/LI-Contextual-Emotions/"
    final_parsed_windows = out_directory + "DesiredWindows.csv"

    reformat_output.parse_script(unfiltered_event_sequence, raw_event_sequence)

    # Function calls in proper order for generation of intermediate files
    facet_event_file.create_facet_event_file(facet_thresh_filename=raw_facet_threshold,
                            facet_event_filename=desired_facet_events,
                            event_filename=raw_event_sequence,
                            only_positives=True,
                            duration_min=0.5)

    merge_event_files.merge_event_files(event_filename=raw_event_sequence,
                      facet_filename=desired_facet_events,
                      event_facet_filename=desired_event_facet,
                      show=False)

    add_action_counts.add_cumulative_action_counts(event_sequence_filename=raw_event_sequence,
                                     event_sequence_cumulatives_output=desired_event_cumulatives_filename)

    windowed_actions.create_windowed_action_file(event_facet_filename=desired_event_facet,
                                   output_filename=desired_windowed_action_filename,
                                   activity_summary_filename=raw_activity_summary,
                                   books_filename=raw_books_and_articles,
                                   window_size=window_size,
                                   omit_AOIs=omit_AOIs,
                                   omit_AUs=omit_AUs,
                                   show=show)

    windowed_actions.parse_windowed_file(windowed_action_file=desired_windowed_action_filename,
                           parsed_windowed_action_file=final_parsed_windows,
                           desired_prefixes=["After-Scan", "After-Submission", "During-Book", "After-Book"],
                           desired_cols=["TestSubject"])