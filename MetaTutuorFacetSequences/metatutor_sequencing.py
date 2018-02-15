import pandas as pd
import numpy as np


# Function for removing AUs, extra emotions, and adding trial number column
def preprocess(facet_event_filename, desired_emotions, remove_AUs=True):
    df = pd.read_csv(facet_event_filename)
    if remove_AUs:
        keep_rows = df.apply(lambda row: "AU" not in row["Target"], axis=1)
        df = df.loc[keep_rows, :]

    if len(desired_emotions) == 0:
        desired_emotions = list(df["Target"].unique())
    else:
        desired_emotions = [e+"Evidence" for e in desired_emotions]

    keep_rows = df.apply(lambda row: row["Target"] in desired_emotions, axis=1)
    df = df.loc[keep_rows, :]
    keep_rows = pd.notna(df["TrialId"])
    df = df.loc[keep_rows, :]
    #df.sort_values(by="TrialId", inplace=True)

    return df


def detect_emotion_number(facet_sequence_file):
    with open(facet_sequence_file, 'r') as seq_file:
        seq_header = seq_file.readline().replace("\n", "").split(",")
        offset = len(seq_header)
        max_key = 0

        for line in seq_file:
            split_line = line.replace("\n", "").split(",")
            try:
                max_sequence_integer = np.array(split_line[offset:], dtype=int).max()
                if max_sequence_integer > max_key:
                    max_key = max_sequence_integer
            except ValueError:
                pass   # A sequence with no observations has occurred, can be ignored here, happens 3 times in MetaTutorIVH2
    return max_key+1, offset  # Plus one since these are 0 based indices but used for length calculations later


def calculate_transitions(int_seq, dim):
    t_matrix = np.zeros((dim, dim))
    for i in range(len(int_seq) - 1):
        t_matrix[int_seq[i], int_seq[i+1]] += 1
    return t_matrix


# Reads in FACET-Sequence-Key.csv and converts into mapping for easy reference to indices
def create_emo_map(emo_map_file):
    return_dict = dict()
    reverse_dict = dict()
    with open(emo_map_file, 'r') as e_file:
        header = e_file.readline()
        for line in e_file:
            split_line = line.replace("\n", "").split(",")
            return_dict[split_line[0]] = int(split_line[1])
            reverse_dict[int(split_line[1])] = split_line[0].replace("Evidence", "")

    return return_dict, reverse_dict


# Function for creating a summary of emotion counts and transitions per trial sequence
def convert_sequence_to_summary(facet_sequence_file, sequence_summary_output, emo_map_file, transitions_of_interest):
    """
    Creates a summary of emotion counts and transitions per trial sequence
    :param facet_sequence_file: filename for the FACET-Sequence.csv
    :param sequence_summary_output: desired filename for the FACET-Sequence-Summary.csv
    :param emo_map_file: filename for the FACET-Sequence-Key.csv used to create map for easy reference
    :param transitions_of_interest: List of pairs of transitions to include in output
    :return: (None) writes results to sequence_summary_output
    """
    emo_map, reverse_map = create_emo_map(emo_map_file)
    emotion_number, offset = detect_emotion_number(facet_sequence_file)
    with open(facet_sequence_file, 'r') as seq_file, open(sequence_summary_output, 'w') as out_file:
        seq_header = seq_file.readline().replace("\n", "")  # Creating header from seq_file to sequence summary
        seq_header += ",Length,"
        seq_header += ",".join(["FACET-"+reverse_map[e] for e in range(emotion_number)]) + ","
        seq_header += ",".join(["FACET-%s-to-%s" % (a.replace("Evidence",""), b.replace("Evidence","")) for a, b in transitions_of_interest])
        out_file.write(seq_header+"\n")

        for line in seq_file:
            split_line = line.replace("\n", "").split(",")
            try:
                integer_sequence = np.array(split_line[offset:], dtype=int)
                length = len(integer_sequence)
                singles = np.bincount(integer_sequence, minlength=emotion_number)
                transition_matrix = calculate_transitions(integer_sequence, dim=emotion_number)
                transition_counts = [transition_matrix[emo_map[a], emo_map[b]] for a, b in transitions_of_interest]
            except ValueError:  # Sequence has zero elements, return 0s for observations and transitions
                length = 0
                singles = np.zeros(emotion_number)
                transition_counts = np.zeros(len(transitions_of_interest))

            # This should mimic the creation of the header so that columns line up with appropriate values
            write_line = ",".join(split_line[:offset])
            write_line += ",%d," % length
            write_line += ",".join([str(e) for e in singles]) + ","
            write_line += ",".join([str(e) for e in transition_counts])
            out_file.write(write_line+"\n")


# Function for creating a sequence file with each contentpage of a trial corresponding to a sequence
def convert_to_trial_sequences(facet_event_filename, facet_sequence_file, facet_key_output, desired_emotions, repeats_allowed=False):
    """
    Creates a sequence file with each content page of a trial corresponding to a sequence (at the student-trial level)
    :param facet_event_filename: String - filename for the FACET-Events file
    :param facet_sequence_file: String - desired filename for the FACET-Sequence output
    :param facet_key_output: String - desired filename for outputting the 
    :param desired_emotions: List of strings containing the emotions that should be tracked, empty list uses all emotions
    :param repeats_allowed: Boolean - removes consecutive emotions of same type if False
    :return: (None) Outputs the FACET-Sequence and FACET-Key files
    """
    facet_df = preprocess(facet_event_filename, desired_emotions=desired_emotions)
    emotion_dictionary = dict()
    count_integer = 0
    prev_subject = ""
    current_sequence = dict()

    with open(facet_sequence_file, 'w') as output_file:
        output_file.write("TestSubject,Instance\n")
        for i,row in facet_df.iterrows():
            if row["TestSubject"] != prev_subject and prev_subject != "":  # Indicates start of new student, writing and resetting data structures
                for j in range(1,19):
                    trial = "%s" % j if j >= 10 else "0%s" % j
                    try:
                        output_file.write("%s,%s,%s\n" % (prev_subject, trial, ",".join(current_sequence[trial])))
                    except KeyError:
                        output_file.write("%s,%s,%s\n" % (prev_subject, trial, ""))
                    current_sequence[trial] = []

            if row["Target"] not in emotion_dictionary.keys():
                emotion_dictionary[row["Target"]] = str(count_integer)
                count_integer += 1
            try:
                trial_number = row["TrialId"][-2:]
                if trial_number not in current_sequence.keys():
                    current_sequence[trial_number] = []
                if repeats_allowed or len(current_sequence[trial_number]) == 0:
                    current_sequence[trial_number].append(emotion_dictionary[row["Target"]])
                elif current_sequence[trial_number][-1] != emotion_dictionary[row["Target"]]:
                    current_sequence[trial_number].append(emotion_dictionary[row["Target"]])
            except TypeError:
                pass
            prev_subject = row["TestSubject"]

        # Writing the last student in the dataset, since no new student after to trigger write
        for j in range(1, 19):
            trial = "%s" % j if j >= 10 else "0%s" % j
            output_file.write("%s,%s,%s\n" % (prev_subject, trial, ",".join(current_sequence[trial])))

    key_df = pd.DataFrame.from_dict(emotion_dictionary,orient='index')
    key_df.columns = ["MappedInteger"]
    key_df.to_csv(facet_key_output)


def create_merged_sequence_summary(facet_sequence_summary, trial_filename, output_filename, desired_trial_cols):
    fss = pd.read_csv(facet_sequence_summary)
    desired_trial_cols = desired_trial_cols + ["TestSubject"] if "TestSubject" not in desired_trial_cols else desired_trial_cols
    desired_trial_cols = desired_trial_cols + ["Instance"] if "Instance" not in desired_trial_cols else desired_trial_cols
    trials = pd.read_csv(trial_filename).loc[:, desired_trial_cols]
    joined_df = fss.merge(trials, on=["TestSubject", "Instance"], how='left', suffixes=['','-Right'])
    joined_df.to_csv(output_filename, index=False)


def create_summary_df(df, t_cols, s_cols, o_cols, other_denom):
    """
    Calculates probabilities of emotions by using number of affect events in trial
    :param df: DataFrame - (aggregated) student-trial summary data
    :param t_cols: List of Strings - columns that are transition columns
    :param s_cols: List of Strings - columns that need to be standardized by Length (for relative probability)
    :param o_cols: List of Strings - columns that need to be standardized by aggregated number
    :param other_denom: Float - Standardizing denominator for o_cols
    :return: DataFrame with standardized values (probabilities) and AffectPerSecond (frequency) column
    """

    df[["%s-Freq" % e for e in s_cols]] = df.loc[:, s_cols].apply(lambda col: col / (df["ContentPage-Duration"] / 60.0),
                                                                  axis=0)
    for t_col in t_cols:
        start_col = t_col[:t_col.index("-to-")]
        df[t_col] = df.loc[:, t_col] / df.loc[:, start_col]
    df[s_cols] = df.loc[:, s_cols].apply(lambda col: col / df["Length"], axis=0)

    for t_col in t_cols:
        end_col = "FACET-" + t_col[t_col.index("-to-") + 4:]
        # Likelihood equation from D'Mello 2012
        df["L-" + t_col] = (df.loc[:, t_col] - df.loc[:, end_col]) / (1 - df.loc[:, end_col])
    df[o_cols] = df.loc[:, o_cols].apply(lambda col: col / other_denom)
    df["AffectPerSecond"] = df["Length"] / df["ContentPage-Duration"]
    df.fillna(value=0, inplace=True)  # Null values created from dividing by 0, for transition probs a denom of 0 should equate to 0
    return df


def create_probability_sequence_summaries(input_filename, base_output, by_trial=True, by_student=True, by_content=True):
    """
    Creates summary useful for analysis from the FACET-Sequences student-trial file
    :param input_filename: String - filename for the FACET-Sequences file
    :param base_output: String - base filename for output of the summaries (suffixes appended for different types)
    :param by_trial: Boolean - Outputs a -Trial summary if True
    :param by_student: Boolean - Outputs a -Student summary if True
    :return: (None) potentially writes multiple summaries
    """
    seq_sum = pd.read_csv(input_filename)
    cols_to_std = [e for e in list(seq_sum.columns.values) if "FACET-" in e and "-to-" not in e]
    trans_cols = [e for e in list(seq_sum.columns.values) if "-to-" in e]
    other_cols = [e for e in list(seq_sum.columns.values) if e not in cols_to_std + trans_cols + ["TestSubject"]]
    if by_trial:
        trial_probs = create_summary_df(seq_sum.copy(), trans_cols, cols_to_std, other_cols, other_denom=1.0)
        trial_probs.to_csv(base_output+"-Trial.csv", index=False)

    if by_student:
        stu_probs = seq_sum.copy().groupby(by="TestSubject").sum()
        stu_probs = create_summary_df(stu_probs, trans_cols, cols_to_std, other_cols, other_denom=18.0)
        stu_probs.to_csv(base_output+"-Student.csv", index=True)

    if by_content:
        num_students = float(len(list(seq_sum["TestSubject"].unique())))
        con_probs = seq_sum.copy().groupby(by="ContentIdNum").sum()
        con_probs = create_summary_df(con_probs, trans_cols, cols_to_std, other_cols, other_denom=num_students)
        con_probs.to_csv(base_output+"-Content.csv", index=True)

if __name__ == "__main__":
    input_filename = "C:/Users/robsc/Documents/NC State/GRAWork/IVHData/OutputSeptember/FACET-ThresholdCrossed/FACET-Events.csv"
    output_filename = "C:/Users/robsc/Documents/NC State/GRAWork/IVHData/OutputSeptember/FACET-ThresholdCrossed/FACET-Sequence.csv"
    summary_output_filename = "C:/Users/robsc/Documents/NC State/GRAWork/IVHData/OutputSeptember/FACET-ThresholdCrossed/FACET-Sequence-Summary.csv"
    key_output_filename = "C:/Users/robsc/Documents/NC State/GRAWork/IVHData/OutputSeptember/FACET-ThresholdCrossed/FACET-Sequence-Key.csv"
    desired_emotions = ["Confusion", "Frustration", "Neutral", "Joy"]  # Default argument for including all emotions, otherwise
    transitions_of_interest = [("ConfusionEvidence", "FrustrationEvidence"),
                              ("ConfusionEvidence", "JoyEvidence"),
                              ("FrustrationEvidence", "JoyEvidence")]

    convert_to_trial_sequences(input_filename, output_filename, key_output_filename, desired_emotions)

    convert_sequence_to_summary(facet_sequence_file=output_filename,
                                sequence_summary_output=summary_output_filename,
                                emo_map_file=key_output_filename,
                                transitions_of_interest=transitions_of_interest)
