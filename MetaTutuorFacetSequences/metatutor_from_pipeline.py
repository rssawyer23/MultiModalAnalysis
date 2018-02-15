from facet_event_file import create_facet_event_file
import metatutor_sequencing as ms
import metatutor_data_preprocessing as mdpp

if __name__ == "__main__":
    # Local system filepaths,
    #   raw = from pipeline,
    #   desired = created by scripts (intermediate or final),
    #   retrieved = not from pipeline but output from somewhere (possible manual or download)
    metatutor_prefix = "C:/Users/robsc/Documents/NC State/GRAWork/IVHData/OutputJan18Neg/"
    metatutor_base_prefix = "C:/Users/robsc/Documents/NC State/GRAWork/IVHData/"

    raw_event_sequence = metatutor_prefix + "EventSequence/EventSequence.csv"
    raw_facet_threshold = metatutor_prefix + "FACET-ThresholdCrossed/FACET-ThresholdCrossed.csv"
    raw_trials_file = metatutor_prefix + "Trials/Trials.csv"
    raw_activity_summary = metatutor_prefix + "ActivitySummary/ActivitySummary.csv"

    desired_facet_events = metatutor_prefix + "FACET-ThresholdCrossed/FACET-Events.csv"
    desired_facet_sequence = metatutor_prefix + "FACET-ThresholdCrossed/FACET-Sequence.csv"
    desired_facet_sequence_key = metatutor_prefix + "FACET-ThresholdCrossed/FACET-Sequence-Key.csv"
    desired_trial_appended = metatutor_prefix + "Trials/Trials_Appended.csv"
    desired_sequence_summary = metatutor_prefix + "FACET-ThresholdCrossed/FACET-Sequence-Summary.csv"
    desired_merged_sequence = metatutor_prefix + "FACET-ThresholdCrossed/FACET-Sequence-Summary-Appended.csv"
    desired_sequence_probability = metatutor_prefix + "FACET-ThresholdCrossed/FACET-Sequence-Summary-Probabilities" #intentionally leaving off .csv
    desired_content_summaries = metatutor_prefix + "Results/Content_Summaries.csv"
    desired_content_key = metatutor_prefix + "Trials/Content_Key.csv"

    retrieved_mc_key = metatutor_base_prefix+"MultipleChoiceKey.csv"
    retrieved_trial_key = metatutor_base_prefix + "IVH_TrialKey.csv"

    # Additional arguments for some of the functions below
    #   Empty list is all emotions for desired emotions argument
    minimum_emotion_duration = 0.5
    minimum_absolute_evidence_score = 0.75
    argument_desired_emotions = ["Joy", "Surprise", "Fear", "Contempt", "Confusion", "Frustration", "Neutral", "Disgust", "Sadness"]
    #argument_desired_emotions = ["Confusion", "Frustration", "Neutral", "Joy", "Surprise"]
    argument_transitions_of_interest = [("ConfusionEvidence", "FrustrationEvidence"),
                                        ("ConfusionEvidence", "JoyEvidence"),
                                        ("FrustrationEvidence", "JoyEvidence"),
                                        ("FrustrationEvidence", "ConfusionEvidence")]
    argument_trial_cols = ["ContentPage-Duration", "MultipleChoiceScores", "MultipleChoiceConfidence",
                           "EaseOfLearning", "JustificationConfidence",
                           "AgentCongruence", "TextRelevancy","DiagramRelevancy",
                           "TextJudgmentScore","DiagramJudgmentScore","PreTestScore", "ContentIdNum"]

    # # Appending useful columns to the trials file through series of functions from preprocessing file
    mdpp.add_content_page_time(event_filename=raw_event_sequence,
                          trial_filename=raw_trials_file,
                          appended_trial_filename=desired_trial_appended)

    mdpp.add_content_numeric_id(trials_filename=desired_trial_appended,
                                trials_output_filename=desired_trial_appended,
                                trial_key_filename=desired_content_key)

    # mdpp.add_fixation_columns(event_sequence_filename=raw_event_sequence,
    #                      trials_filename=desired_trial_appended,
    #                      output_filename=desired_trial_appended)

    mdpp.grade_multiple_choice(mc_key_filename=retrieved_mc_key,
                          trials_filename=desired_trial_appended,
                          output_filename=desired_trial_appended,
                          content_output_name=desired_content_summaries)

    mdpp.grade_relevancies(trial_key_filename=retrieved_trial_key,
                          trials_filename=desired_trial_appended,
                          output_filename=desired_trial_appended)

    mdpp.add_pre_test(activity_summary_filename=raw_activity_summary,
                 trials_filename=desired_trial_appended,
                 output_filename=desired_trial_appended)

    # Use facet threshold file to create a FACET-Event file
    create_facet_event_file(facet_thresh_filename=raw_facet_threshold,
                            facet_event_filename=desired_facet_events,
                            event_filename=raw_event_sequence,
                            only_positives=True,
                            duration_min=minimum_emotion_duration,
                            absolute_min=minimum_absolute_evidence_score)
    #
    # Use the FACET-Event file to create trial sequences of affect
    ms.convert_to_trial_sequences(facet_event_filename=desired_facet_events,
                               facet_sequence_file=desired_facet_sequence,
                               facet_key_output=desired_facet_sequence_key,
                               desired_emotions=argument_desired_emotions)

    # Use the trial sequences of affect to create summaries/counts of each trial's sequence
    ms.convert_sequence_to_summary(facet_sequence_file=desired_facet_sequence,
                                sequence_summary_output=desired_sequence_summary,
                                emo_map_file=desired_facet_sequence_key,
                                transitions_of_interest=argument_transitions_of_interest)

    ms.create_merged_sequence_summary(facet_sequence_summary=desired_sequence_summary,
                                   trial_filename=desired_trial_appended,
                                   output_filename=desired_merged_sequence,
                                   desired_trial_cols=argument_trial_cols)

    ms.create_probability_sequence_summaries(input_filename=desired_merged_sequence,
                                          base_output=desired_sequence_probability,
                                          by_trial=True, by_student=True)

