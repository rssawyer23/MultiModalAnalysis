import pandas as pd
import numpy as np
import datetime
from dateutil import parser


AOIS_OF_INTEREST = ["VirtualHuman", "ContentText", "FigureImage"]
REMOVE_TEST_SUBJECTS = ["IVH2_PN031", "IVH2_PN065"]


def determine_AOI_targets(event_filename="EventSequence.csv", desired_page_name="ContentPage"):
    """
    Function for getting the headers to use in the dictionary for AOIs
    :param event_filename: 
    :param desired_page_name: 
    :return targets: list of potential AOI targets
    """
    data = pd.read_csv(event_filename)
    page_rows = data.loc[:,"PageName"] == desired_page_name
    targets = list(data[page_rows,"Target"].unique())
    return targets


def initialize_headers(header_line):
    """
    Function for creating a dictionary of header:index pairs
    :param header_line: The first line of a file that contains comma separated column headers
    :return: dictionary mapping header names to indexes of a csv split array
    """
    headers = dict()
    header_split = header_line.split(sep=",")
    for header_index in range(len(header_split)):
        headers[header_split[header_index].replace("\n","")] = header_index
    return headers


def format_previous_trial_data(test_subject, trial_id, data_increments, page_type="ContentPage"):
    """
    Converting the data increment dictionary to a list format for appending to trial fixation list
    :param test_subject: TestSubject identification number
    :param trial_id: The corresponding trial number for the fixations
    :param data_increments: The dictionary with the AOIs of interest plus total fixation duration
    :return: list of information that can be easily converted into a dataframe
    """
    return_array = [test_subject, trial_id]
    return_array.append(data_increments["%s-Time" % page_type])
    return_array.append(data_increments["%sFixation-Total" % page_type])
    return_array.append(data_increments["%sFixation-Count" % page_type])

    for aoi in AOIS_OF_INTEREST:
        return_array.append(data_increments[aoi])
    return_array.append(data_increments["Other-Total"])
    for aoi in AOIS_OF_INTEREST:
        try:
            return_array.append(data_increments[aoi] / data_increments["%sFixation-Total" % page_type])
        except ZeroDivisionError:
            return_array.append(0)
    try:
        return_array.append(data_increments["Other-Total"] / data_increments["%sFixation-Total" % page_type])
    except ZeroDivisionError:
        return_array.append(0)
    for aoi in AOIS_OF_INTEREST:
        return_array.append(data_increments["%s-Count" % aoi])
    return_array.append(data_increments["Other-Count"])
    return return_array


def calculate_fixations(event_file, e_h, page_type="ContentPage"):
    """
    Reading through the EventSequence file to increment AOI durations of different targets for a specific page
    :param event_file: 
    :param e_h: Abbreviation for event_headers dictionary
    :return trial_fixations: dataframe with fixations appended
        Column headers will be: TestSubject, TrialId, ContentPageDuration, fixation times, fixation proportions columns
    """
    # For each trial, increment durations until end of content page, then calculate proportions
    reset = True
    prev_test_subject = "NO PREVIOUS SUBJECT"
    prev_trial_id = "NO PREVIOUS TRIAL ID"
    start_set, end_set = False, False
    start_page, end_page = datetime.datetime.now(), datetime.datetime.now()
    data_increments = dict()
    trial_fixations = []
    line_number = 1
    for line in event_file:
        # Reading in the next line
        split = line.split(",")
        if len(split) == len(e_h.keys()):
            test_subject = split[e_h["TestSubject"]]
            trial_id = split[e_h["TrialId"]]

            # New trial detected, resetting the dictionary
            if prev_trial_id != trial_id or prev_test_subject != test_subject:
                if prev_test_subject != "NO PREVIOUS SUBJECT" and len(prev_trial_id) > 2:
                    data_increments["%s-Time" % page_type] = (end_page - start_page).total_seconds()
                    trial_fixations.append(format_previous_trial_data(prev_test_subject, prev_trial_id, data_increments))
                # add previous info/data increments to trial fixations if not initial
                start_set, end_set = False, False
                data_increments = {"%s-Time" % page_type: None,
                                   "%sFixation-Total" % page_type: 0,
                                   "%sFixation-Count" % page_type: 0,
                                   "Other-Total":0,
                                   "Other-Proportion":0,
                                   "Other-Count":0}
                for aoi in AOIS_OF_INTEREST:
                    data_increments[aoi] = 0
                    data_increments["%s-Count" % aoi] = 0
                reset = False
            # Handling a valid row of AOI on desired page type
            if split[e_h["PageName"]] == page_type and split[e_h["Event"]] == "AOI":
                if not start_set:
                    start_page = parser.parse(split[e_h["TimeStamp"]])
                    start_set = True
                # Currently discarding AOIs not in the AOIs of interest
                data_increments["%sFixation-Total" % page_type] += float(split[e_h["Duration"]])
                data_increments["%sFixation-Count" % page_type] += 1
                if split[e_h["Target"]] in data_increments.keys():
                    data_increments[split[e_h["Target"]]] += float(split[e_h["Duration"]])
                    data_increments["%s-Count" % split[e_h["Target"]]] += 1
                else:
                    data_increments["Other-Total"] += float(split[e_h["Duration"]])
                    data_increments["Other-Count"] += 1
            elif split[e_h["PageName"]] != page_type and start_set and not end_set:
                end_page = parser.parse(split[e_h["TimeStamp"]])
                end_set = True
            elif not reset:
                reset = True
            else:
                pass
            prev_test_subject = test_subject
            prev_trial_id = trial_id
            line_number += 1
    data_increments["%s-Time" % page_type] = (end_page - start_page).total_seconds()
    trial_fixations.append(format_previous_trial_data(prev_test_subject, prev_trial_id, data_increments))
    return trial_fixations


def generate_fixation_header(page_type="ContentPage"):
    header = ["TestSubject", "TrialId", "%s-Time" % page_type, "%sFixation-Total" % page_type, "%sFixation-Count" % page_type]
    for aoi in AOIS_OF_INTEREST:
        header.append("%s-Total" % aoi)
    header.append("Other-Total")
    for aoi in AOIS_OF_INTEREST:
        header.append("%s-Proportion" % aoi)
    header.append("Other-Proportion")
    for aoi in AOIS_OF_INTEREST:
        header.append("%s-Count" % aoi)
    header.append("Other-Count")
    return header


def merge_fixations(trial_df, fixation_df):
    for student_id in REMOVE_TEST_SUBJECTS:
        fixation_df = fixation_df.loc[fixation_df["TestSubject"] != student_id, :]
        trial_df = trial_df.loc[trial_df["TestSubject"] != student_id, :]
    full_data = trial_df.join(other=fixation_df, rsuffix="-f")
    return full_data


def add_fixation_columns(event_sequence_filename="EventSequence.csv", trials_filename="Trials.csv",
                         output_filename="Trials_wFixations.csv"):
    """
    Function for adding time fixating and proportion fixating on items from content page
    :param event_sequence_filename: filename of the event sequence for MetaTutor data
        Columns needed: TestSubject, Event, Duration, TrialId, PageName, Target
    :param trials_filename: filename of the trials to append columns to
        Columns needed: TestSubject, TrialId
    :param output_filename: filename to output the appended dataframe to
    :return: dataframe that is the trials.csv file with columns appended for the fixations and proportion fixating
    """

    event_file = open(file=event_sequence_filename, mode='r')
    event_headers = initialize_headers(event_file.readline())
    start = datetime.datetime.now()
    print("Starting fixation calculations %s..." % start)
    trial_fixations = calculate_fixations(event_file, event_headers)
    fixation_header = generate_fixation_header()
    fixation_df = pd.DataFrame(trial_fixations,columns=fixation_header)
    end = datetime.datetime.now()
    print("Fixations calculated from EventSequence %s" % end)
    print("Total seconds elapsed %.5f" % (end - start).total_seconds())
    trial_data = pd.read_csv(trials_filename)
    full_data = merge_fixations(trial_data, fixation_df)
    full_data.to_csv(output_filename, index=False)


def output_content_summaries(partial_dictionary, output_filename="Content_Summaries.csv"):
    """
    Function for outputting diagnostics for each of the ContentIds for debugging and difficulty assessment
    :param partial_dictionary: Dictionary mapping ContentId to list with correct, partial1, partial2, incorrect
    :param output_filename: String of filename for destination of content output
    :return: True if completed successfully
    """
    output_file = open(output_filename, 'w')
    header = "ContentId,Correct,Partial1,Partial2,Incorrect,WeightedScore,PercentCorrect,TotalAttempted,Ungraded\n"
    output_file.write(header)
    for key in partial_dictionary.keys():
        total_attempted = partial_dictionary[key][0] + partial_dictionary[key][1] + partial_dictionary[key][2] + partial_dictionary[key][3]
        weighted_score = partial_dictionary[key][0] + 0.5 * partial_dictionary[key][1] + 0.5 * partial_dictionary[key][2]
        try:
            percent_correct = partial_dictionary[key][0] / float(total_attempted)
        except ZeroDivisionError:
            percent_correct = 0
        write_line = "%s,%d,%d,%d,%d,%.5f,%.5f,%d,%d" % (key, partial_dictionary[key][0], partial_dictionary[key][1],
                                                      partial_dictionary[key][2], partial_dictionary[key][3],
                                                      weighted_score, percent_correct, total_attempted,
                                                         partial_dictionary[key][4])
        output_file.write(write_line+"\n")
    output_file.close()
    return True


def add_content_numeric_id(trials_filename, trials_output_filename, trial_key_filename):
    trial_data = pd.read_csv(trials_filename)
    categorical = pd.Categorical(trial_data["ContentId"])
    trial_data["ContentIdNum"] = categorical.codes
    trial_data.to_csv(trials_output_filename, index=False)

    key_df = pd.DataFrame()
    key_df["ContentId"] = categorical.categories
    key_df["ContentIdNum"] = pd.Series(range(18))
    key_df.to_csv(trial_key_filename, index=False)


def grade_multiple_choice(mc_key_filename="", trials_filename="Trials.csv", output_filename="Trials_wFixations.csv", content_output_name="Content_Summaries.csv"):
    """
    Function for adding 0, 0.5, 1 scores for the multiple choice questions among trials
    :param mc_key_filename: string for filename of the multiple choice answer key (columns: ContentId, Correct, Partial1, Partial2)
    :param trials_filename: string for filename fo the trials
    :param output_filename: string for the desired output location of the trials dataframe with multiple choice answers
    :return: returns true if completed successfully
    """
    # Load data
    answer_key = pd.read_csv(mc_key_filename)
    trial_data = pd.read_csv(trials_filename)
    answer_scores = []

    # iterate through rows of trial file, checking multiple choice response for accuracy against multiple choice key
    count_p1, count_p2 = 0, 0
    partial_dictionary = dict()
    for index, row in trial_data.iterrows():
        answer_row = answer_key.loc[answer_key["ContentId"] == row["ContentId"]]

        if row["ContentId"] not in partial_dictionary.keys():
            partial_dictionary[row["ContentId"]] = [0,0,0,0,0]

        if answer_row.shape[0] != 1:
            print("ContentId Error: %s in row %d" % (row["ContentId"], index))
            return False
        answer_row = answer_row.iloc[0, :]
        if answer_row["Correct"].strip() == row["MultipleChoiceResponse"].strip():
            partial_dictionary[row["ContentId"]][0] += 1
            answer_scores.append(1)
        elif answer_row["Partial1"].strip() == row["MultipleChoiceResponse"].strip():
            partial_dictionary[row["ContentId"]][1] += 1
            count_p1 += 1
            answer_scores.append(0.5)
        elif answer_row["Partial2"].strip() == row["MultipleChoiceResponse"].strip():
            partial_dictionary[row["ContentId"]][2] += 1
            count_p2 += 1
            answer_scores.append(0.5)
        elif answer_row["Incorrect"].strip() == row["MultipleChoiceResponse"].strip():
            partial_dictionary[row["ContentId"]][3] += 1
            answer_scores.append(0)
        else:
            partial_dictionary[row["ContentId"]][4] += 1
            answer_scores.append(-1)

    output_content_summaries(partial_dictionary, content_output_name)


    # Add column for scores and output result
    trial_data["MultipleChoiceScores"] = pd.Series(answer_scores,dtype=float)
    trial_data.to_csv(output_filename, index=False)
    return True


def grade_relevancies(trial_key_filename="IVH_TrialKey.csv", trials_filename="Trials.csv", output_filename="Trials_wFixations.csv"):
    trial_key = pd.read_csv(trial_key_filename)
    trial_data = pd.read_csv(trials_filename)
    additional_columns = []
    additional_column_headers = ["AgentCongruence", "TextRelevancy", "DiagramRelevancy", "TextJudgmentScore", "DiagramJudgmentScore"]

    for index, row in trial_data.iterrows():
        add_row = []
        key_row = trial_key.loc[trial_key["ContentId"] == row["ContentId"]]
        if key_row.shape[0] != 1:
            print("ContentId Error: %s in row %d" % (row["ContentId"], index))
            return False
        key_row = key_row.iloc[0, :]

        # Adding Agent congruency; -1, 0, or 1 for Incongruent, Neutral, and Congruent Agent expressions
        if key_row["Congruency"] == "C":
            add_row.append(1)
        elif key_row["Congruency"] == "I":
            add_row.append(-1)
        else:
            add_row.append(0)

        # Adding text and diagram relevencies
        if key_row["Relevancy"] == "TLR": # Text Less Relevant; append 0 for text, 1 for diagram
            add_row.append(0)
            add_row.append(1)
        elif key_row["Relevancy"] == "DLR": # Diagram Less Relevant; append 1 for text, 0 for diagram
            add_row.append(1)
            add_row.append(0)
        else: # == "AR" # All Relevant; append 1 for text, 1 for diagram
            add_row.append(1)
            add_row.append(1)

        # Adding text correctness
        if row["TextJudgmentResponse"] == "TextSomewhatRelevent":
            add_row.append(0.5)
        elif add_row[1] == 0 and row["TextJudgmentResponse"] == "TextNotRelevent" or add_row[1] == 1 and row["TextJudgmentResponse"] == "TextRelevent":
            add_row.append(1)
        else:
            add_row.append(0)

        # Adding diagram correctness
        if row["DiagramJudgmentResponse"] == "DiagramSomewhatRelevent":
            add_row.append(0.5)
        elif add_row[2] == 0 and row["DiagramJudgmentResponse"] == "DiagramNotRelevent" or add_row[2] == 1 and row[
                "DiagramJudgmentResponse"] == "DiagramRelevent":
            add_row.append(1)
        else:
            add_row.append(0)

        additional_columns.append(add_row)
    additional_df = pd.DataFrame(additional_columns, columns=additional_column_headers)
    trial_data = trial_data.join(additional_df)
    trial_data.to_csv(output_filename, index=False)


def add_pre_test(activity_summary_filename="ActivitySummary.csv", trials_filename="Trials_wFixations.csv", output_filename="Trials_wExtras"):
    act_sum = pd.read_csv(activity_summary_filename)
    map_id = pd.Series(act_sum.PreTestScore.values,index=act_sum.TestSubject).to_dict()
    trials = pd.read_csv(trials_filename)
    trials["PreTestScore"] = trials.apply(lambda row: map_id[row["TestSubject"]], axis=1)
    trials.to_csv(output_filename, index=False)


def add_student_content_summary(trials_filename="../Output/Results/Trials_wExtras.csv",
                                output_filename="../Output/Results/Content_Summaries_Appended.csv"):
    data = pd.read_csv(trials_filename)
    num_students = len(data["TestSubject"].unique())
    content_data = data.groupby(by="ContentId")["EaseOfLearning","MultipleChoiceScores","ContentPage-Time",
                                                "TextJudgmentScore","DiagramJudgmentScore","TextRelevancy", "DiagramRelevancy",
                                                "AgentCongruence","Stu-StdEaseOfLearning","Stu-EoLError","Stu-DirectionCorrect", "Stu-BaseError"].mean()
    content_data["AgentCongruence"] = content_data["AgentCongruence"] + 1
    content_data["MultipleChoiceScores"] = content_data["MultipleChoiceScores"] * num_students
    content_data["StdScores"] = (content_data["MultipleChoiceScores"] - content_data["MultipleChoiceScores"].mean())/content_data["MultipleChoiceScores"].std()

    content_data.to_csv(output_filename, index=True)  # Index is the ContentId


def generate_error_features_df(data, error="squared"):
    # Calculating difficulty of each trial type
    num_students = len(data["TestSubject"].unique())
    content_data = data.groupby(by="ContentId")["EaseOfLearning", "MultipleChoiceScores", "ContentPage-Time",
                                                "TextJudgmentScore", "DiagramJudgmentScore", "TextRelevancy", "DiagramRelevancy",
                                                "AgentCongruence", "Stu-StdEaseOfLearning", "Stu-EoLError", "Stu-DirectionCorrect", "Stu-BaseError"].mean()
    content_data["AgentCongruence"] = content_data["AgentCongruence"] + 1
    content_data["MultipleChoiceScores"] = content_data["MultipleChoiceScores"] * num_students
    content_data["StdScores"] = (content_data["MultipleChoiceScores"] - content_data["MultipleChoiceScores"].mean()) / content_data["MultipleChoiceScores"].std()

    # Calculating error based on difficulty calculated from the training fold
    map_id = pd.Series(content_data.StdScores.values, index=content_data.index).to_dict()
    data["StdEaseOfTrial"] = data.apply(lambda row: map_id[row["ContentId"]], axis=1)
    data["StdEaseOfLearning"] = (data["EaseOfLearning"] - data["EaseOfLearning"].mean()) / data["EaseOfLearning"].std()

    data["Stu-StdEaseOfLearning"] = np.zeros(data.shape[0])
    for stu_id in data["TestSubject"].unique():
        student_rows = data["TestSubject"] == stu_id
        data.loc[student_rows, "Stu-StdEaseOfLearning"] = (data.loc[student_rows, "EaseOfLearning"] - data.loc[
            student_rows, "EaseOfLearning"].mean()) / data.loc[student_rows, "EaseOfLearning"].std()
    if error == "squared":
        data["EoLError"] = (data["StdEaseOfLearning"] - data["StdEaseOfTrial"]) ** 2
        data["Stu-EoLError"] = (data["Stu-StdEaseOfLearning"] - data["StdEaseOfTrial"]) ** 2
    else:  # Assuming absolute error
        data["EoLError"] = np.absolute(data["StdEaseOfLearning"] - data["StdEaseOfTrial"])
        data["Stu-EoLError"] = np.absolute(data["Stu-StdEaseOfLearning"] - data["StdEaseOfTrial"])
    data["Stu-BaseError"] = data["Stu-StdEaseOfLearning"] - data["StdEaseOfTrial"]
    data["DirectionCorrect"] = pd.Series(
        np.logical_or(np.equal(data["StdEaseOfLearning"] > 0, data["StdEaseOfTrial"] > 0),
                      np.equal(data["StdEaseOfLearning"] < 0, data["StdEaseOfTrial"] < 0)), dtype=int)
    data["Stu-DirectionCorrect"] = pd.Series(
        np.logical_or(np.equal(data["Stu-StdEaseOfLearning"] > 0, data["StdEaseOfTrial"] > 0),
                      np.equal(data["Stu-StdEaseOfLearning"] < 0, data["StdEaseOfTrial"] < 0)), dtype=int)
    return data


def add_eol_error(trials_filename, content_filename, output_filename, error="squared"):
    data = pd.read_csv(trials_filename)
    content_data = pd.read_csv(content_filename)
    map_id = pd.Series(content_data.StdScores.values,index=content_data.ContentId).to_dict()
    data["StdEaseOfTrial"] = data.apply(lambda row: map_id[row["ContentId"]], axis=1)
    data["StdEaseOfLearning"] = (data["EaseOfLearning"] - data["EaseOfLearning"].mean())/data["EaseOfLearning"].std()

    data["Stu-StdEaseOfLearning"] = np.zeros(data.shape[0])
    for stu_id in data["TestSubject"].unique():
        student_rows = data["TestSubject"] == stu_id
        data.loc[student_rows, "Stu-StdEaseOfLearning"] = (data.loc[student_rows, "EaseOfLearning"] - data.loc[student_rows, "EaseOfLearning"].mean()) / data.loc[student_rows, "EaseOfLearning"].std()
    if error == "squared":
        data["EoLError"] = (data["StdEaseOfLearning"] - data["StdEaseOfTrial"])**2
        data["Stu-EoLError"] = (data["Stu-StdEaseOfLearning"] - data["StdEaseOfTrial"])**2
    else:  # Assuming absolute error
        data["EoLError"] = np.absolute(data["StdEaseOfLearning"] - data["StdEaseOfTrial"])
        data["Stu-EoLError"] = np.absolute(data["Stu-StdEaseOfLearning"] - data["StdEaseOfTrial"])
    data["Stu-BaseError"] = data["Stu-StdEaseOfLearning"] - data["StdEaseOfTrial"]
    data["DirectionCorrect"] = pd.Series(np.logical_or(np.equal(data["StdEaseOfLearning"] > 0, data["StdEaseOfTrial"] > 0),
                                             np.equal(data["StdEaseOfLearning"] < 0, data["StdEaseOfTrial"] < 0)), dtype=int)
    data["Stu-DirectionCorrect"] = pd.Series(np.logical_or(np.equal(data["Stu-StdEaseOfLearning"] > 0, data["StdEaseOfTrial"] > 0),
                                             np.equal(data["Stu-StdEaseOfLearning"] < 0, data["StdEaseOfTrial"] < 0)), dtype=int)
    data.to_csv(trials_filename,index=False)  # THIS OVERWRITES AN INPUT FILE
    student_eol_df = data.groupby(by="TestSubject")["EaseOfLearning", "EoLError","MultipleChoiceScores","TextJudgmentScore",
                                                    "DiagramJudgmentScore","PreTestScore","ContentPage-Time",
                                                    "DirectionCorrect","StdEaseOfLearning", "Stu-StdEaseOfLearning",
                                                    "Stu-EoLError","Stu-DirectionCorrect","Stu-BaseError"].mean() # "Stu-StdEaseOfLearning" should all be 0s
    student_eol_df.to_csv(output_filename, index=True)  # Index is the StudentId (TestSubject)


def sequential_error_summary(trials_filename="../Output/Results/Trials_wExtras.csv",
                             output_filename="../Output/Results/Sequential_Trial_Summary.csv"):
    data = pd.read_csv(trials_filename)
    number_students = len(data["TestSubject"].unique())
    trialn_data = data.groupby(by="Instance")["Stu-StdEaseOfLearning", "Stu-EoLError", "Stu-DirectionCorrect"].mean()
    trialn_data["Stu-EoLErrorStdDev"] = data.groupby(by="Instance")["Stu-EoLError"].std()
    trialn_data["Stu-EolErrorStdErr"] = data.groupby(by="Instance")["Stu-EoLError"].std() / np.sqrt(number_students)
    trialn_data.to_csv(output_filename, index=True)


def add_gender_to_student_summary(student_filename, pretest_filename):
    ss_data = pd.read_csv(student_filename)
    pt_data = pd.read_csv(pretest_filename,header=0)
    pt_data["NumericGender"] = pt_data.apply(lambda row: 0 if row["Gender "][0].upper() == "F" else 1, axis=1)  # FEMALES ARE 0, MALES ARE 1
    merged = pd.merge(left=ss_data, right=pt_data, how='inner', left_on=["TestSubject"], right_on=["username"])
    merged.drop(labels=["Gender ","username"], axis=1, inplace=True)
    merged.to_csv(student_filename, index=False)  # This overwrites the input file


def add_content_page_time(event_filename, trial_filename, appended_trial_filename):
    event_df = pd.read_csv(event_filename)
    keep_rows = event_df["PageName"] == "ContentPage"
    event_df = event_df.loc[keep_rows,:]

    content_min = pd.to_datetime(event_df.groupby("TrialId")["TimeStamp"].min())
    content_max = pd.to_datetime(event_df.groupby("TrialId")["TimeStamp"].max())

    content_time = content_max - content_min
    content_time = content_time.apply(lambda cell: cell.total_seconds())
    content_time = pd.DataFrame(content_time)
    content_time.columns = ["ContentPage-Duration"]

    trials = pd.read_csv(trial_filename)

    merged = trials.merge(content_time, how='left', left_on='TrialId', right_index=True)
    merged.to_csv(appended_trial_filename, index=False)


# REQUIRES FILES:
#   EventSequence.csv
#   Trials.csv
#   MultipleChoiceKey.csv
#   IVH_TrialKey.csv
# PRODUCES FILES:
#   Trials_wExtras.csv
#   Content_Summaries.csv
#   Content_Summaries_Averaged.csv
if __name__ == "__main__":
    prefix = "C:/Users/robsc/Documents/NC State/GRAWork/IVHData/OutputSeptember/"
    ivh_prefix = "C:/Users/robsc/Documents/NC State/GRAWork/IVHData/"
    add_content_page_time(event_filename=prefix+"EventSequence/EventSequence.csv",
                          trial_filename=prefix+"Trials/Trials.csv",
                          appended_trial_filename=prefix+"Trials/Trials_wExtras.csv")

    add_fixation_columns(event_sequence_filename=prefix+"EventSequence/EventSequence.csv",
                         trials_filename=prefix+"Trials/Trials_wExtras.csv",
                         output_filename=prefix+"Trials/Trials_wExtras.csv")

    grade_multiple_choice(mc_key_filename=ivh_prefix+"MultipleChoiceKey.csv",
                          trials_filename=prefix+"Trials/Trials_wExtras.csv",
                          output_filename=prefix+"Trials/Trials_wExtras.csv",
                          content_output_name=prefix+"Results/Content_Summaries.csv")

    grade_relevancies(trial_key_filename=ivh_prefix+"IVH_TrialKey.csv",
                          trials_filename=prefix+"Trials/Trials_wExtras.csv",
                          output_filename=prefix+"Trials/Trials_wExtras.csv")

    add_pre_test(activity_summary_filename=prefix+"ActivitySummary/ActivitySummary.csv",
                 trials_filename=prefix+"Trials/Trials_wExtras.csv",
                 output_filename=prefix+"Results/Trials_wExtras.csv")

    add_eol_error(trials_filename=prefix+"Results/Trials_wExtras.csv",
                  content_filename=prefix+"Results/Content_Summaries_Averaged.csv",
                  output_filename=prefix+"Results/Student_Summaries.csv",
                  error="squared")

    add_gender_to_student_summary(student_filename=prefix+"Results/Student_Summaries.csv",
                                  pretest_filename=ivh_prefix+"IVH2_GenderKey.csv")

    add_student_content_summary(trials_filename=prefix+"Results/Trials_wExtras.csv",
                                output_filename=prefix+"Results/Content_Summaries_Averaged.csv")

    sequential_error_summary(trials_filename=prefix+"Results/Trials_wExtras.csv",
                             output_filename=prefix+"Results/Sequential_Trial_Summary.csv")
