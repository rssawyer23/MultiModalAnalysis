import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from dateutil.parser import parse
import datetime
import matplotlib.patches as mpatches


def _expand_sequence(seq, extra):
    """Helper function for padding a numpy array (seq) with (extra) amount of rows duplicating the last row"""
    last_row = seq[-1, :]
    extra_data = np.tile(last_row, (extra, 1))
    full = np.concatenate([seq, extra_data])
    return full


def knn_prediction(sorted_series, k, response_data, response_variable, predict_type="equal", regularize_prop=0.1, pass_logistic=False):
    """Predict (response_variable) using the (k) closest subjects by distance from (sorted_series)
        regularize_prop = proportion of prediction to allocate to average of all subjects, 
            i.e. 0 is no regularization, 1 is using average as predictor (not using knn prediction)"""
    all_subjects = list(sorted_series.index)
    average_response_rows = response_data.loc[:, "TestSubject"].apply(lambda cell: cell in all_subjects)
    average_response = response_data.loc[average_response_rows, response_variable].mean()

    closest_subjects = list(sorted_series.index[:k])
    response_rows = response_data.loc[:, "TestSubject"].apply(lambda cell: cell in closest_subjects)
    if predict_type == "equal":
        prediction = response_data.loc[response_rows, response_variable].mean()
    elif predict_type == "weighted":
        indexed_rows = response_data.loc[response_rows, response_variable]
        indexed_rows.index = response_data.loc[response_rows, "TestSubject"]
        inv_distances = 1 / sorted_series.loc[closest_subjects]
        distance_proportions = (inv_distances / np.sum(inv_distances))
        prediction = np.sum(indexed_rows * distance_proportions)
    else:
        print("Unknown prediction type")
        prediction = 0.0

    # Prediction as weighted sum of KNN prediction with average prediction
    regularized_prediction = (1.0 - regularize_prop) * prediction + regularize_prop * average_response
    if pass_logistic:
        regularized_prediction = 2.0/(1 + np.exp(-1*regularized_prediction)) - 1.0
    return regularized_prediction


def calculate_error(prediction, subject, response_data, response_variable):
    subject_row = response_data.loc[:, "TestSubject"] == subject
    if np.sum(subject_row) != 1:
        print("Error finding data for %s" % subject)
        return 0.0
    true_value = response_data.loc[subject_row, response_variable].iloc[0]
    return true_value - prediction


def generate_time_series_indices(time_stamps, total_seconds, interval=1):
    """Passing the TimeStamps of a student's events dataframe and total seconds, generate sequence indices for distance calculations"""
    start_time = parse(time_stamps.iloc[0])
    time_intervals = [start_time + datetime.timedelta(seconds=i) for i in range(int(total_seconds / interval))]
    sequence_indices = []
    sequence_counter = 0
    for i in range(len(time_intervals)):
        while sequence_counter < len(time_stamps) and parse(time_stamps[sequence_counter + 1]) < time_intervals[i]:
            sequence_counter += 1
        sequence_indices.append(sequence_counter)
    return sequence_indices


def determine_seconds(one_times, two_times, length_mismatch):
    """Helper function that uses differences in times to determine total seconds that need to be discretized into indices"""
    seq_one_time = (parse(one_times["TimeStamp"].iloc[-1]) - parse(one_times["TimeStamp"].iloc[0])).total_seconds()
    seq_two_time = (parse(two_times["TimeStamp"].iloc[-1]) - parse(two_times["TimeStamp"].iloc[0])).total_seconds()
    if length_mismatch == "minimum":
        seconds = min(seq_one_time, seq_two_time)
    elif length_mismatch == "padded":
        seconds = max(seq_one_time, seq_two_time)
    else:
        print("Undefined handler for length mismatch: %s" % length_mismatch)
        seconds = 0
    return seconds


def series_distance(seq_one, seq_two, distance_features, length_mismatch="minimum"):
    """Calculating the distance using the timestamps instead of sequence index"""
    total_seconds = determine_seconds(seq_one["TimeStamp"], seq_two["TimeStamp"], length_mismatch)
    seq_one_indices = generate_time_series_indices(time_stamps=seq_one["TimeStamp"], total_seconds=total_seconds)
    seq_two_indices = generate_time_series_indices(time_stamps=seq_two["TimeStamp"], total_seconds=total_seconds)
    assert len(seq_one_indices) == len(seq_two_indices)

    reduced_one = np.array(seq_one.loc[:, distance_features])
    reduced_two = np.array(seq_two.loc[:, distance_features])
    distances = []
    for i in range(len(seq_one_indices)):
        distance_at_time = np.linalg.norm(reduced_one[seq_one_indices[i], :] - reduced_two[seq_two_indices[i], :])
        distances.append(distance_at_time)
    return np.mean(distances), np.std(distances), np.max(distances)


def sequence_distance(seq_one, seq_two, length_mismatch="minimum"):
    """Calculating distances using the sequence index, with two methods of handling length mismatches"""
    if length_mismatch == "minimum":
        # Truncate the longer sequence so same indices are compared
        final_index = min(seq_one.shape[0], seq_two.shape[0])
        reduced_one = np.array(seq_one.iloc[:final_index, :])
        reduced_two = np.array(seq_two.iloc[:final_index, :])
    elif length_mismatch == "padded":
        # Pad the length of the shorter sequence by repeating last element
        length_difference = seq_one.shape[0] - seq_two.shape[0]
        if length_difference < 0:
            reduced_one = _expand_sequence(np.array(seq_one), abs(length_difference))
            reduced_two = np.array(seq_two)
        else:
            reduced_one = np.array(seq_one)
            reduced_two = _expand_sequence(np.array(seq_two), abs(length_difference))

    else:
        print("Undefined length mismatch handler: %s" % length_mismatch)
        return -1.0, -1.0

    interval_distances = np.apply_along_axis(lambda row: np.linalg.norm(row), axis=1, arr=(reduced_one - reduced_two))
    avg_dist = np.mean(interval_distances)
    std_dist = np.std(interval_distances)
    max_dist = np.max(interval_distances)
    return avg_dist, std_dist, max_dist


def get_subject_distance(data, subject_one, subject_two, distance_features, time_based=False, length_mismatch="minimum"):
    one_rows = data.loc[:, "TestSubject"] == subject_one
    one_data = data.loc[one_rows, :]
    two_rows = data.loc[:, "TestSubject"] == subject_two
    two_data = data.loc[two_rows, :]
    if not time_based:
        avg_dist, std_dist, max_dist = sequence_distance(seq_one=one_data.loc[:, distance_features],
                                                         seq_two=two_data.loc[:, distance_features],
                                                         length_mismatch=length_mismatch)
    else:
        avg_dist, std_dist, max_dist = series_distance(seq_one=one_data,
                                                       seq_two=two_data,
                                                       distance_features=distance_features,
                                                       length_mismatch=length_mismatch)
    return avg_dist, std_dist, max_dist


def create_dist_dict(data, distance_features, omit_list, time_based=False, length_mismatch="minimum", show=False):
    # TODO Pickle this so it does not need to be recreated each time, possibly include metadata for arguments
    """Create a 'distance dictionary' which maps pairs of test subjects to distance, variance pairs
            This is essentially a distance matrix represented by a dictionary for indexing purposes"""
    start_time = datetime.datetime.now()
    print("Started creating distance dictionary: %s" % start_time)
    subjects = [e for e in list(data["TestSubject"].unique()) if e not in omit_list]
    dist_dict = dict()
    for outer_index in range(len(subjects)):
        for inner_index in range(outer_index, len(subjects)):
            dist, std, max_d = get_subject_distance(data=data,
                                                    subject_one=subjects[outer_index],
                                                    subject_two=subjects[inner_index],
                                                    distance_features=distance_features,
                                                    time_based=time_based,
                                                    length_mismatch=length_mismatch)
            dist_dict[(subjects[outer_index], subjects[inner_index])] = {"Distance":dist, "Std":std, "Max":max_d}
            dist_dict[(subjects[inner_index], subjects[outer_index])] = {"Distance":dist, "Std":std, "Max":max_d}
    if show:
        print("Finished creating distance dictionary in %.4f minutes" % ((datetime.datetime.now() - start_time).total_seconds()/60.0))
    return dist_dict


def subject_distances_series(dist_dict, subject):
    """Using the distance dictionary, create a sorted series of the distance values"""
    dist_list = []
    name_list = []
    for key in dist_dict.keys():
        if key[0] == subject and key[1] != subject:
            dist_list.append(dist_dict[key]["Distance"])
            name_list.append(key[1])
    dist_series = pd.Series(dist_list, index=name_list)
    dist_series.sort_values(inplace=True)
    return dist_series


def _determine_features(event_columns, features):
    """Functionality for handling:
            None: use all cumulatives
            len=1: use only cumulatives of specific type (i.e. 'A' actions or 'L' for locations)
            list: use the list as specific names for the columns"""
    if features is None:
        return_list = [e for e in event_columns if "C-" in e if "Posters" not in e]
    elif len(features) == 1:
        return_list = [e for e in event_columns if "C-%s-" % features in e if "Posters" not in e]
    else:
        return_list = features
    return return_list


def fit_pca(pca_obj, event_df, feature_list, omit_list):
    """Function for getting the PCA object trained from the final row of all students
            Use omit_list to omit students from PCA training for proper cross-validation procedure"""

    subjects = [e for e in list(event_df.loc[:, "TestSubject"].unique()) if e not in omit_list and pd.notna(e)]
    subject_df = pd.DataFrame()
    for subject in subjects:
        subject_rows = event_df.loc[:, "TestSubject"] == subject
        full_subject_rows = event_df.loc[subject_rows, feature_list]
        last_subject_row = full_subject_rows.iloc[full_subject_rows.shape[0]-1, :]
        subject_df = pd.concat([subject_df, last_subject_row], axis=1)
    subject_df = subject_df.T

    # subject_df.index = subjects
    # subject_df.to_csv("C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/ActivitySummary/ActivitySummaryCumulatives.csv", index=True)

    scaler = StandardScaler()
    pca_obj.fit(scaler.fit_transform(subject_df))
    return scaler


def condense_data(subject_data, length=100):
    interval = (subject_data.shape[0]-1) / float(length)
    index_list = [int((i+1) * interval) for i in range(length)]
    return subject_data.iloc[index_list, :]


def _get_prop_value(color_val, max_color_val, min_color_prop, reverse_fill=True):
    remaining_proportion = (1 - min_color_prop)
    color_fill_proportion = abs(color_val / max_color_val)
    if reverse_fill:
        return_color = (1 - color_fill_proportion) * remaining_proportion
    else:
        return_color = color_fill_proportion * remaining_proportion
    return return_color + min_color_prop


def _get_color(subject, add_data, color_by, max_color_by_val=1, min_color_prop=0.6):
    """Function for getting an RGB tuple to be used as trajectory color, determined from color_by"""
    add_rows = add_data.loc[:, "TestSubject"] == subject
    if np.sum(add_rows) != 1:
        print("Invalid Add Data for %s" % subject)
        return tuple([0.0, 0.0, 0.0])
    else:
        subject_color_val = add_data.loc[add_rows, color_by].iloc[0]
        prop_val = _get_prop_value(color_val=subject_color_val, max_color_val=max_color_by_val,
                                   min_color_prop=min_color_prop, reverse_fill=True)
        if subject_color_val > 0:
            color_tuple = tuple([0.0, prop_val, 0.0])
        elif subject_color_val < 0:
            color_tuple = tuple([prop_val, 0.0, 0.0])
        else:
            color_tuple = tuple([0.0,0.0,0.0])
        return color_tuple


def _get_legend_handles(golden_present):
    lowest = mpatches.Patch(color=tuple([_get_prop_value(-1.0, 1.0, 0.6, True), 0.0, 0.0]), label="Lowest NLG")
    low = mpatches.Patch(color=tuple([_get_prop_value(-0.4, 1.0, 0.6, True), 0.0, 0.0]), label="Low NLG")
    highest = mpatches.Patch(color=tuple([0.0, _get_prop_value(1.0, 1.0, 0.6, True), 0.0]), label="Highest NLG")
    high = mpatches.Patch(color=tuple([0.0, _get_prop_value(0.4, 1.0, 0.6, True), 0.0]), label="High NLG")
    none = mpatches.Patch(color=tuple([0.0,0.0,0.0]), label="No Learning Gain")
    gold = mpatches.Patch(color=tuple([0.83, 0.88, 0.13]), label="Golden Path")
    if golden_present:
        return [highest, high, none, low, lowest, gold]
    else:
        return [highest, high, none, low, lowest]


def plot_trajectories(x, y, data, add_data, color_by, omit_list, golden, save_file):
    """Function for plotting trajectories using the x and y as names of data to plot and color_by for shading the lines"""
    fig, ax = plt.subplots(1)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title("Crystal Island Student Trajectories")
    subjects = [e for e in list(data.loc[:, "TestSubject"].unique()) if e not in omit_list and pd.notna(e)]
    invalid_count = 0
    for subject in subjects:
        subject_rows = data.loc[:, "TestSubject"] == subject
        subject_data = data.loc[subject_rows, :]
        if subject_data.shape[0] > 10:
            subject_start_time = parse(subject_data.loc[:, "TimeStamp"].iloc[0])
            subject_data["DurationElapsed"] = subject_data.loc[:, "TimeStamp"].apply(lambda cell: (parse(cell) - subject_start_time).total_seconds())
            plot_x = np.array(subject_data.loc[:, x])
            plot_y = np.array(subject_data.loc[:, y])
            color = _get_color(subject, add_data, color_by)
            ax.plot(plot_x, plot_y, color=color, linestyle="solid", linewidth=0.5)
        else:
            invalid_count += 1

    if x in golden.columns and y in golden.columns:
        golden_present = True
        plot_x = np.array(golden.loc[:, x])
        plot_y = np.array(golden.loc[:, y])
        gold_color = tuple([0.83, 0.88, 0.13])
        ax.plot(plot_x, plot_y, color=gold_color, linestyle="solid", linewidth=1.0)
    else:
        golden_present = False

    legend_handles = _get_legend_handles(golden_present)
    ax.legend(handles=legend_handles, loc=4)

    print("Plotted %d Trajectories" % (len(subjects) - invalid_count))
    fig.savefig(save_file)
    plt.close()


def get_reference_path_distances(reference_df, other_df, omit_list, distance_features, length_mismatch):
    start_time = datetime.datetime.now()
    print("Started creating distance dictionary: %s" % start_time)
    subjects = [e for e in list(other_df["TestSubject"].unique()) if e not in omit_list]
    dist_df = pd.DataFrame()
    for subject in subjects:
        subject_rows = other_df.loc[:, "TestSubject"] == subject
        subject_data = other_df.loc[subject_rows, distance_features]
        subj_avg, subj_std, subj_max = sequence_distance(seq_one=reference_df.loc[:, distance_features],
                                                 seq_two=subject_data,
                                                 length_mismatch=length_mismatch)  # Returns (average, std) tuple
        dist_df[subject] = pd.Series([subj_avg, subj_std, subj_max])
    dist_df = dist_df.T
    dist_df.index = subjects
    dist_df.columns = ["Average", "Std", "Max"]
    return dist_df

if __name__ == "__main__":
    directory = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/"
    event_seq_filename = directory + "EventSequence/EventSequence_wCGS_NoAOI.csv"
    act_sum_filename = directory + "ActivitySummary/ActivitySummaryGraded.csv"
    image_output_filename = directory + "ActivitySummary/Trajectories.png"
    golden_events_filename = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/EventSequence/GoldenPathEventSequence_wC.csv"
    golden_distance_output = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/EventSequence/GoldenPathDistances.csv"

    golden_events = pd.read_csv(golden_events_filename)
    act_sum = pd.read_csv(act_sum_filename).loc[:, ["TestSubject", "NLG", "FinalGameScore", "Post-Presence", "LearningGain", "Duration"]]

    action_features = ["C-A-Conversation", "C-A-BooksAndArticles",
                         "C-A-Worksheet", "C-A-PlotPoint", "C-A-WorksheetSubmit",
                         "C-A-Scanner"]
    location_features = ["C-L-Dock", "C-L-Beach", "C-L-Outside",
                         "C-L-Infirmary", "C-L-Lab", "C-L-Dining", "C-L-Dorm", "C-L-Bryce"]
    full_features = action_features + location_features

    predict_arguments = {"Response":"NLG",
                         "K":5,
                         "Predict Type":"weighted",
                         "Distance Features":action_features,
                         "Distance Predict Features":["PC1"],
                         "Distance Mismatch":"padded",
                         "Distance Time Type":True,
                         "Regularization":0.0}

    events = pd.read_csv(event_seq_filename)
    keep_rows = np.logical_and(pd.notnull(events["TimeStamp"]), pd.notnull(events["TestSubject"]))
    events = events.loc[keep_rows, :]

    non_full_omit_list = [e for e in list(act_sum["TestSubject"].unique()) if "1301" not in e]
    features = predict_arguments["Distance Features"]

    feature_list = _determine_features(list(events.columns.values), features)
    components = 4
    pca_transformer = PCA(n_components=components)
    scaler = fit_pca(pca_obj=pca_transformer, event_df=events, feature_list=feature_list, omit_list=non_full_omit_list)
    print(np.cumsum(pca_transformer.explained_variance_ratio_)[:components])
    print(pd.DataFrame([pca_transformer.components_[0], pca_transformer.components_[1]], columns=feature_list).T)

    transformed_gp = pca_transformer.transform(golden_events.loc[:, feature_list].fillna(0))
    transformed_gp_df = pd.DataFrame(transformed_gp, columns=["PC%d" % (n+1) for n in range(transformed_gp.shape[1])], index=list(range(golden_events.shape[0])))
    golden_events.index = list(range(golden_events.shape[0]))
    full_golden_df = pd.concat([golden_events, transformed_gp_df], axis=1)

    transformed_matrix = pca_transformer.transform(events.loc[:, feature_list].fillna(0))
    transformed_df = pd.DataFrame(transformed_matrix, columns=["PC%d" % (n+1) for n in range(transformed_matrix.shape[1])], index=list(range(events.shape[0])))
    events.index = list(range(events.shape[0]))
    full_df = pd.concat([events, transformed_df], axis=1)

    # Calculating distances with reference path and outputting dataframe after appending NLG
    # gold_distances = get_reference_path_distances(full_golden_df, full_df,
    #                                               omit_list=non_full_omit_list,
    #                                               distance_features=predict_arguments["Distance Predict Features"],
    #                                               length_mismatch=predict_arguments["Distance Mismatch"])
    # nlg_df = act_sum.loc[:, ["TestSubject", "NLG", "Duration"]]
    # nlg_df.index = nlg_df.loc[:, "TestSubject"]
    # nlg_df.drop(labels="TestSubject", inplace=True, axis=1)
    # merged_distances = gold_distances.merge(right=nlg_df, how='left', left_index=True, right_index=True)
    # merged_distances["TestSubject"] = merged_distances.index
    # merged_distances.to_csv(golden_distance_output, index=False)

    # print(get_subject_distance(full_df,
    #                            subject_one="CI1301PN008",
    #                            subject_two="CI1301PN006",
    #                            distance_features=["PC1", "PC2"],
    #                            length_mismatch="minimum"))

    # plot_trajectories(x="DurationElapsed", y="PC1",
    #                   data=full_df, add_data=act_sum,
    #                   color_by="NLG",
    #                   omit_list=non_full_omit_list,
    #                   golden=full_golden_df,
    #                   save_file=image_output_filename)

    #distance_features = ["PC1", "PC2", "PC3", "PC4"]

    full_df.loc[:, predict_arguments["Distance Features"]] = scaler.transform(X=full_df.loc[:, predict_arguments["Distance Features"]])
    d = create_dist_dict(full_df,
                           distance_features=predict_arguments["Distance Predict Features"],
                           omit_list=non_full_omit_list,
                           time_based=predict_arguments["Distance Time Type"],
                           length_mismatch=predict_arguments["Distance Mismatch"])

    all_errors = []
    full_subject_list = [e for e in list(act_sum["TestSubject"].unique()) if "1301" in e]
    full_subject_list.remove("CI1301PN042")  # No Pre/Post Data
    full_subject_list.remove("CI1301PN043")  # No Pre/Post Data

    for subject in full_subject_list:
        s = subject_distances_series(d, subject=subject)

        prediction = knn_prediction(s, k=predict_arguments["K"],
                                    response_data=act_sum,
                                    response_variable=predict_arguments["Response"],
                                    predict_type=predict_arguments["Predict Type"],
                                    regularize_prop=predict_arguments["Regularization"])
        # print("Prediction for %s: %.4f" % (subject, prediction))
        error = calculate_error(prediction=prediction,
                                subject=subject,
                                response_data=act_sum,
                                response_variable=predict_arguments["Response"])
        if pd.notnull(error):
            all_errors.append(error)
        # desired_subject = subject
        # keep_rows = act_sum.loc[:, "TestSubject"] == desired_subject
        # print(act_sum.loc[keep_rows, :])
        # print("Error for %s: %.4f" % (subject, error))
    print("\n-------------------------FINISHED PREDICTING %s---------------------------------" % predict_arguments["Response"])
    full_subject_rows = act_sum.loc[:,"TestSubject"].apply(lambda sub: sub in full_subject_list)
    mean_errors = act_sum.loc[full_subject_rows, predict_arguments["Response"]] - act_sum.loc[full_subject_rows, predict_arguments["Response"]].mean()
    print("Predicted: %d\tTotal: %d" % (len(all_errors), np.sum(full_subject_rows)))
    print("MAE: %.4f\tmMAE: %.4f" % (np.mean(np.abs(all_errors)), np.mean(np.abs(mean_errors))))
    print("MSE: %.4f\tmMSE: %.4f" % ((np.mean(np.array(all_errors)**2)), (np.mean(np.array(mean_errors)**2))))
    rss = np.sum(np.array(all_errors)**2)
    tss = np.sum(mean_errors**2)
    print("RSS: %.4f" % rss)
    print("TSS: %.4f" % tss)
    print("R2: %.4f" % (1.0 - rss/tss))
