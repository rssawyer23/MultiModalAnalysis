import pandas as pd
import grade_qualtrics as gq
import REFLECT.clean_responses as cr
import add_action_counts as aac
import reformat_output as ro
import add_game_score as ags
import add_interval_durations as aid

if __name__ == "__main__":
    show = False
    reflect_directory = "C:/Users/robsc/Documents/NC State/GRAWork/CI-REFLECT/REFLECT-2-2018/Output/"
    leads_directory = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output318/"

    raw_reflect_act_sum = reflect_directory + "ActivitySummary/ActivitySummary.csv"
    raw_reflect_event_sequence = reflect_directory + "EventSequence/EventSequence.csv"
    raw_revised_reflect_event_sequence = reflect_directory + "EventSequence/EventSequenceP.csv"
    raw_reflect_books_articles = reflect_directory + "BooksAndArticles/BooksAndArticles.csv"

    desired_reflect_graded_act_sum = reflect_directory + "ActivitySummary/ActivitySummaryGraded.csv"
    desired_reflect_event_sequence = reflect_directory + "EventSequence/EventSequenceCumulatives.csv"
    desired_reflect_graded_act_sum_std = reflect_directory + "ActivitySummary/ActivitySummaryGradedStd.csv"
    desired_reflect_appended_act_sum = reflect_directory + "ActivitySummary/ActivitySummaryAppended.csv"
    desired_reflect_edited_act_sum = reflect_directory + "ActivitySummary/ActivitySummaryAppendedEdited.csv"

    retrieved_grading_key = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/SurveyGradingScheme.csv"
    retrieved_post_test = "C:/Users/robsc/Documents/NC State/GRAWork/CI-REFLECT/REFLECT-2-2018/Reflect-Post-Non-Numeric.csv"
    retrieved_pre_test= "C:/Users/robsc/Documents/NC State/GRAWork/CI-REFLECT/REFLECT-2-2018/Reflect-Pre-Non-Numeric.csv"

    gq.grade_tests(grading_key_filename=retrieved_grading_key,
                   post_test_filename=retrieved_post_test,
                   pre_test_filename=retrieved_pre_test,
                   activity_summary_filename=raw_reflect_act_sum,
                   output_filename=desired_reflect_graded_act_sum,
                   normalize=True, show=show)

    cr.clean_responses_to_numeric(activity_summary_inpath=desired_reflect_graded_act_sum,
                                  activity_summary_outpath=desired_reflect_graded_act_sum)

    ro.parse_script(input_filename=raw_reflect_event_sequence,
                    output_filename=raw_revised_reflect_event_sequence)

    ags.add_game_score(event_filename=raw_revised_reflect_event_sequence,
                   activity_summary=desired_reflect_graded_act_sum,
                   books_articles=raw_reflect_books_articles,
                   appended_event_filename=desired_reflect_event_sequence,
                   appended_activity_summary=desired_reflect_graded_act_sum)

    aac.add_cumulative_action_counts(event_sequence_filename=desired_reflect_event_sequence,
                                     event_sequence_cumulatives_output=desired_reflect_event_sequence)

    aac.append_cumulative_action_counts(event_sequence_filename=desired_reflect_event_sequence,
                                        activity_summary_filename=desired_reflect_graded_act_sum,
                                        appended_activity_summary_filename=desired_reflect_appended_act_sum)
    cr.clean_student_rows(appended_activity_summary_filename=desired_reflect_appended_act_sum,
                          edited_activity_summary_filename=desired_reflect_edited_act_sum)

    # get leads activity summary
