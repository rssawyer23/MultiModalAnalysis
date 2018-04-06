# Script for performing affect analysis data on Crystal Island Full Agency students for UMUAI 3-15-2018
import pandas as pd
import os.path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from parse_events import cumulative_event_blocks
import datetime
from error_struct import ErrorStruct
import statsmodels.api as sm
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix


def remove_null_facet_rows(df):
    """Remove students with null facet data as determined by examining the mean evidence score of a facet column"""
    non_null_facet_rows = pd.notnull(data["MeanJoyEvidence"])
    df = df.loc[non_null_facet_rows, :]
    df.index = list(range(df.shape[0]))
    return df


def remove_null_trace_rows(df):
    """Remove students with null trace data as determined by null or 0 scores in Final Game Score"""
    non_null_trace_rows = pd.notnull(data["FinalGameScore"]) & data["FinalGameScore"] != 0
    df = df.loc[non_null_trace_rows, :]
    df.index = list(range(df.shape[0]))
    return df


def get_data_point_color(cell):
    """Simple color selection method to give green (positive), black (0) or red (negative) learning gains"""
    if cell < 0:
        color_tuple = tuple([1.0, 0.0, 0.0])
    else:
        color_tuple = tuple([0.0, 1.0, 0.0])
    return color_tuple


def clean_col(col):
    inf_indices = col == np.inf
    col.loc[inf_indices] = 0
    return col


def get_title_label(plot_filename):
    if "Base" in plot_filename:
        return_label = "Base"
    elif "Emotion" in plot_filename:
        return_label = "Emotions"
    elif "AU" in plot_filename:
        return_label = "Action Units"
    else:
        return_label = 'Undefined'
    return return_label


def apply_fitted_transformation(df, columns, pca, scaler, standardize_by="GameTime"):
    """Use already fitted pca scaler and standard scaler on training data to apply to test data"""
    X = df.loc[:, columns]
    omit_list = ["GameTime"]
    if standardize_by != "":
        duration_std_X = pd.DataFrame()
        for col in X.columns:
            duration_std_X[col] = X.loc[:, col] / df.loc[:, standardize_by] if col not in omit_list else X.loc[:, col]
        duration_std_X = duration_std_X.apply(clean_col, axis=0)
        duration_std_X.fillna(0, inplace=True)
    else:
        duration_std_X = X

    scaled_X = scaler.transform(duration_std_X)
    reduced_X = pca.transform(scaled_X)
    return pd.DataFrame(reduced_X, columns=["PC%d" % (c + 1) for c in range(pca.n_components_)])


def get_scaled_data(df, X, standardize_by, omit_list):
    standard_scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    if standardize_by != "":
        duration_std_X = pd.DataFrame()
        for col in X.columns:
            duration_std_X[col] = X.loc[:, col] / df.loc[:, standardize_by] if col not in omit_list else X.loc[:, col]
        # duration_std_X = X.apply(lambda col: col / df.loc[:, standardize_by], axis=0)
        duration_std_X = duration_std_X.apply(clean_col, axis=0)
        duration_std_X.fillna(0, inplace=True)
    else:
        duration_std_X = X
    scaled_X = standard_scaler.fit_transform(duration_std_X)
    return scaled_X, standard_scaler


def pca_analysis(df, columns, components, output_filename="", plot_output_filename="", standardize_by="GameTime", color_by="NLG", show=False, plot=True):
    """Use PCA to transform the data to a smaller dimension to determine the underlying variance of the data
        Returns the data reduced to number of dimensions specific by (components)"""
    pca_transformer = PCA(n_components=components)
    omit_list=["GameTime"]
    X = df.loc[:, columns]

    scaled_X, standard_scaler = get_scaled_data(df, X, standardize_by=standardize_by, omit_list=omit_list)
    reduced_X = pca_transformer.fit_transform(scaled_X)
    colors = (df[color_by] - response_median).apply(get_data_point_color)  # REPLACE THIS FUNCTION
    # colors = standard_scaler.fit_transform(df[color_by])

    component_variance = pd.Series(np.cumsum(pca_transformer.explained_variance_ratio_),
                                   index=["PC%d" % (c+1) for c in range(components)])
    component_matrix = pd.DataFrame(pca_transformer.components_,
                                    index=["PC%d" % (c+1) for c in range(components)],
                                    columns=columns)
    component_matrix["CumulativeComponentVariance"] = component_variance
    component_matrix = component_matrix.T
    if show:
        print(component_matrix)
    if len(output_filename) > 1:
        # print((duration_std_X*60).mean())
        # print((duration_std_X*60).std())
        component_matrix.to_csv(output_filename, index=True)

    if len(plot_output_filename) > 1:
        fig, ax = plt.subplots(1)
        ax.scatter(x=reduced_X[:, 0], y=reduced_X[:, 1], c=colors)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        title_label = get_title_label(plot_output_filename)
        ax.set_title("First Two Factor Scores of %s Colored by %s" % (title_label, color_by))
        high = mpatches.Patch(color=tuple([0.0, 1.0, 0.0]), label="High %s Class" % color_by)
        low = mpatches.Patch(color=tuple([1.0, 0.0, 0.0]), label="Low %s Class" % color_by)
        legend_handles = [high, low]
        ax.legend(handles=legend_handles, loc=4)
        plt.savefig(plot_output_filename)
    if plot:
        plt.show()
    plt.close()

    return pd.DataFrame(reduced_X, columns=["PC%d" % (c+1) for c in range(components)]), pca_transformer, standard_scaler


def _reduce(e):
    """Get the specific emotion/AU of the column by removing excess information from interval and type"""
    try:
        start = e.index("-")
        end = e.index("Evidence")
        word = e[(start+1):end]
        return word
    except ValueError:
        return "STRING NOT IN PROPER FORMAT FOR OPERATION"


def get_columns(all_cols, interval, type, emotes):
    red_cols = [e for e in all_cols if "%s-" % interval in e and "-%s" % type in e]
    if emotes == "Emotions":
        final_cols = [e for e in red_cols if "Evidence" in e and "AU" not in e]
    elif emotes == "AUs":
        final_cols = [e for e in red_cols if "Evidence" in e and "AU" in e]
    else:
        final_cols = [e for e in red_cols if "Evidence" in e and _reduce(e) in emotes]
    return final_cols


def initialize_error_dictionary(names):
    ed = dict()
    ed["MeanRegressor"] = []
    for name in names:
        ed[name] = dict()
        ed[name]["Base"] = []
        ed[name]["wAUs"] = []
        ed[name]["wEmotions"] = []
    return ed


def output_time(task_name, start):
    current_time = datetime.datetime.now()
    seconds = (current_time - start).total_seconds()
    print("Finished %s in %.4f seconds" % (task_name, seconds))
    return current_time


def get_last_subject_rows(timed_data):
    all_subjects = list(timed_data["TestSubject"].unique())
    last_row_df = pd.DataFrame()
    for subject in all_subjects:
        subject_rows = timed_data["TestSubject"] == subject
        subject_df = timed_data.loc[subject_rows, :]
        last_subject_row = subject_df.iloc[-1, :]
        last_row_df = pd.concat([last_row_df, last_subject_row], axis=1)
    last_row_df = last_row_df.T
    last_row_df.index = list(range(last_row_df.shape[0]))
    return last_row_df


def calculate_variance_inflation_factors(df):
    """Calculate the variance inflation factor for each column in the dataframe and return an indexed series of the VIFs"""
    vifs = pd.Series(index=list(df.columns))
    for col in df.columns:
        feature_list = [e for e in df.columns if e != col]
        lm = LinearRegression()
        lm.fit(X=df.loc[:,feature_list], y=df.loc[:,col])
        r2 = lm.score(X=df.loc[:,feature_list], y=df.loc[:,col])
        vifs[col] = 1.0 / (1.0 - r2)
    return vifs


def log_reg_final_output(data, features, pca_output, plot_output, a_output, response="NLG"):
    class_series = data.loc[:, response] >= response_median
    print(class_series.describe())
    response_class = np.array(class_series, dtype=int)
    reduced_data, pca, scale = pca_analysis(data, output_filename=pca_output,
                                            plot_output_filename=plot_output,
                                            columns=features, components=6, color_by="NLG", show=False, plot=False)
    scaled_data, scaler = get_scaled_data(df=data, X=data.loc[:, features],
                                  standardize_by="GameTime", omit_list=["GameTime"])

    og_columns = reduced_data.shape[1]
    final_SVC = SVC(kernel='linear')
    final_SVC.fit(X=reduced_data, y=response_class)
    N = np.array(scaled_data)
    print("N:" + str(N.shape))
    P = pca.components_
    print(np.array(P).dot(np.array(N).T)[0,:])
    print("P:" + str(P.shape))
    w = final_SVC.coef_
    print("w:" + str(w.shape))
    b = final_SVC.intercept_
    print("b:" + str(b.shape))
    A = np.array(w).dot(np.array(P))
    print("A:" + str(A.shape))
    pd.DataFrame(A.T, index=features, columns=["Magnitude"]).to_csv(a_output)
    distances = (np.array(A).dot(np.array(N).T) + np.array(b)).T
    print(distances.shape)

    print("%s SVM Boundary" % pca_output)
    print("Intercept %.5f" % final_SVC.intercept_)
    print(pd.Series(final_SVC.coef_[0, :], index=["PC%s" % (i+1) for i in range(og_columns)]))
    reduced_data["Distance"] = pd.Series(final_SVC.decision_function(X=reduced_data))
    reduced_data["PredClass"] = reduced_data["Distance"] > 0
    reduced_data["Class"] = response_class
    reduced_data["ManualDistances"] = distances
    print(reduced_data)
    print(confusion_matrix(y_true=reduced_data["Class"], y_pred=reduced_data["PredClass"]))
    reduced_data.to_csv("C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output318/Results/FinalSVMDF.csv", index=False)


    # fig, ax = plt.subplots(1)
    # ax.scatter(x=reduced_data.loc[:,"PC1"], y=reduced_data.loc["PC2"])

    logit = sm.Logit(response_class, sm.add_constant(np.asarray(reduced_data.loc[:, ["PC%s" % (i+1) for i in range(og_columns)]])))
    result = logit.fit()
    print(result.summary())


def plot_SVM_boundary(data, features, pca_output, plot_output, boundary_output, f1=0, f2=1, response="NLG"):
    class_series = data.loc[:, response] < response_median
    print(class_series.describe())
    response_class = np.array(class_series, dtype=int)
    reduced_data, pca, scale = pca_analysis(data, output_filename=pca_output,
                                            plot_output_filename=plot_output,
                                            columns=features, components=6, color_by="NLG", show=False, plot=False)
    last_four_averages = reduced_data.mean(axis=0).iloc[2:]
    final_SVC = SVC(kernel='linear')
    final_SVC.fit(X=reduced_data, y=response_class)

    fig, ax = plt.subplots(1)
    ax.scatter(reduced_data.iloc[:, f1], reduced_data.iloc[:, f2], c=class_series.apply(lambda cell: tuple([1.0, 0, 0]) if cell else tuple([0, 1.0, 0])))
    ax.set_xlabel("PC%d" % (f1+1))
    ax.set_ylabel("PC%d" % (f2+1))
    ax.set_title("Linear SVM Boundary")

    # create grid to evaluate model
    xx = np.linspace(reduced_data.iloc[:, f1].min(), reduced_data.iloc[:, f1].max(), 30)
    yy = np.linspace(reduced_data.iloc[:, f2].min(), reduced_data.iloc[:, f2].max(), 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.zeros((900, 6))
    xy[:, f1] = XX.ravel().T
    xy[:, f2] = YY.ravel().T
    print(final_SVC.coef_)
    Z = final_SVC.decision_function(xy).reshape(XX.shape)

    # Plot decision boundary (w margins) and support vectors
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    ax.scatter(final_SVC.support_vectors_[:, f1], final_SVC.support_vectors_[:, f2], s=100, linewidth=1, facecolors='none')

    # Create and plot Legend
    high = mpatches.Patch(color=tuple([0.0, 1.0, 0.0]), label="High NLG Class")
    low = mpatches.Patch(color=tuple([1.0, 0.0, 0.0]), label="Low NLG Class")
    legend_handles = [high, low]
    ax.legend(handles=legend_handles, loc=4)

    plt.show()
    fig.savefig(boundary_output[:-4] + "F%dF%d.png" % (f1, f2))


def output_correlation_files(data, response, feature_dict, output_filenames, show=False):
    for key in feature_dict.keys():
        temp_df = pd.DataFrame(data.loc[:, [response] + feature_dict[key]], dtype=float).corr()
        temp_df["VIFs"] = calculate_variance_inflation_factors(data.loc[:, feature_dict[key]])
        temp_df.to_csv(output_filenames[key], index=True)
        if show:
            print(temp_df)


def run_cross_validation():
    pass

def print_accuracy_results():
    pass

if __name__ == "__main__":
    results_directory = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output318/Results/"
    output_filenames = {"BaseCorr":"%sBaseFeaturesCorrelation.csv" % results_directory,
                        "EmotionCorr":"%sEmotionFeaturesCorrelation.csv" % results_directory,
                        "AUCorr":"%sAUFeaturesCorrelation.csv" % results_directory,
                        "AllCorr":"%sAllFeatureCorrelation.csv" % results_directory,
                        "BasePCA":"%sBaseFeaturesPCA.csv" % results_directory,
                        "EmotionPCA":"%sEmotionFeaturesPCA.csv" % results_directory,
                        "AUPCA":"%sAUFeaturesPCA.csv" % results_directory,
                        "AllPCA":"%sAllFeaturesPCA.csv" % results_directory,
                        "BasePCAPlot": "%sBaseFeaturesPCAPlot.png" % results_directory,
                        "EmotionPCAPlot": "%sEmotionFeaturesPCAPlot.png" % results_directory,
                        "AUPCAPlot": "%sAUFeaturesPCAPlot.png" % results_directory,
                        "AllPCAPlot":"%sAllFeaturesPCAPlot.png" % results_directory,
                        "SVMBoundary":"%sSVMBoundary.png" % results_directory,
                        "SVMPCA_Transform":"%sSVMPCA_Transform.csv" % results_directory
                        }
    interval_predictions = False
    start_time = datetime.datetime.now()
    data_filename = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output318/ActivitySummary/ActivitySummaryAppended.csv"
    data = pd.read_csv(data_filename)
    data = remove_null_facet_rows(data)
    data = remove_null_trace_rows(data)
    last_time = output_time("Reading Activity Summary", start_time)

    response_column = "NLG"  # Typically one of [NLG, Post-Presence, FinalGameScore/CumulativeGameScore]
    response_median = data[response_column].median()


    # Adding the time-interval based data for use in modelling
    data_save_filename = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output318/ActivitySummary/SupervisedLearningIntervals2.csv"
    if os.path.isfile(data_save_filename):
        timed_data = pd.read_csv(data_save_filename)
    else:
        timed_data, counts = cumulative_event_blocks(event_filename="C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output318/EventSequence/EventSequenceFACET.csv",
                                activity_filename="C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output318/ActivitySummary/ActivitySummaryAppended.csv",
                                interval=5*60,
                                add_columns=["NLG", "Post-Presence", "PreTestScore"],
                                keep_list=[e for e in list(data["TestSubject"].unique()) if e != "CI1301PN066"])
        timed_data.to_csv(data_save_filename, index=False)
    last_time = output_time("Reading Event Examples", last_time)

    keep_rows = timed_data["TestSubject"] != "CI1301PN066"
    timed_data = timed_data.loc[keep_rows, :]
    timed_data.index = list(range(timed_data.shape[0]))

    emotion_features = [e for e in list(timed_data.columns) if "C-FD-" in e and "AU" not in e
                        and "Positive" not in e and "Negative" not in e and "Neutral" not in e]
    au_features = [e for e in list(timed_data.columns) if "C-FD-" in e and "AU" in e and "AU7" not in e and "AU5" not in e]
    base_features = [e for e in list(timed_data.columns) if "C-A-" in e
                     and "Movement" not in e and "FACET" not in e and "Posters" not in e and "PlotPoint" not in e] + \
                    ["C-D-Conversation", "C-D-BooksAndArticles", "C-D-Worksheet"]
    all_features = base_features + emotion_features + au_features

    final_data = get_last_subject_rows(timed_data)
    feature_dict = {"BaseCorr":base_features,
                    "EmotionCorr":emotion_features,
                    "AUCorr":au_features,
                    "AllCorr":all_features}
    output_correlation_files(data=final_data, response="NLG",
                             feature_dict=feature_dict, output_filenames=output_filenames,
                             show=False)

    # USE THE DF WITH 900+ OBSERVATIONS IF TRUE, OTHERWISE USE THE DF WITH 58
    if not interval_predictions:
        timed_data = final_data
    #FINAL ONLY DATAFRAME CREATION HERE timed_data = final_data_rows
    #
    response_variables = ["NLG", "Post-Presence", "CumulativeGameScore"]
    multimodal_handler = "Additive"  # Can be one of ["Channels", "Specific", "Additive"]

    # Should loop through a range of IntervalID to get cumulative models for each time stamp (train using all in LOOCV?)

    if multimodal_handler == "Channels":
        reduced_emotions, pca_emotions, scale_emotions = pca_analysis(timed_data, columns=emotion_features, components=2, color_by="NLG", show=True, plot=False)
        reduced_aus, pca_aus, scale_aus = pca_analysis(timed_data, columns=au_features, components=2, color_by="NLG", show=True, plot=False)
        reduced_base, pca_base, scale_pca = pca_analysis(timed_data, columns=base_features, components=3, color_by="NLG", show=True, plot=False)

        # Creating supersets of features for Base + AUs and Base + Emotions
        reduced_aus = pd.concat([reduced_base, reduced_aus], axis=1)
        reduced_emotions = pd.concat([reduced_base, reduced_emotions], axis=1)
        reduced_all = pd.DataFrame()
    elif multimodal_handler == "Specific":
        reduced_emotions, pca_emotions, scale_emotions = pca_analysis(timed_data, columns=["C-FD-JoyEvidence", "C-FD-FrustrationEvidence"],
                                                                      components=1, color_by="NLG", show=True,
                                                                      plot=False, standardize_by="")
        reduced_aus, pca_aus, scale_aus = pca_analysis(timed_data, columns=["C-FD-AU4Evidence"], components=1, color_by="NLG",
                                                       show=True, plot=False, standardize_by="")
        reduced_base, pca_base, scale_pca = pca_analysis(timed_data, columns=base_features, components=3,
                                                         color_by="NLG", show=True, plot=False)

        # Creating supersets of features for Base + AUs and Base + Emotions
        reduced_aus = pd.concat([reduced_base, reduced_aus], axis=1)
        reduced_emotions = pd.concat([reduced_base, reduced_emotions], axis=1)
        reduced_all = pd.DataFrame()
    else:  # multimodal_handler == "Additive"
        reduced_base, pca_base, scale_pca = pca_analysis(timed_data, output_filename="",
                                                         plot_output_filename="",
                                                         columns=base_features, components=6, color_by="NLG",
                                                         show=True, plot=False)
        reduced_emotions, pca_emotions, scale_emotions = pca_analysis(timed_data, output_filename="",
                                                                      plot_output_filename="",
                                                         columns=base_features + emotion_features, components=6,
                                                         color_by="NLG", show=True, plot=False)
        reduced_aus, pca_aus, scale_aus = pca_analysis(timed_data, output_filename="",
                                                       plot_output_filename="",
                                                       columns=base_features + au_features, components=6,
                                                       color_by="NLG", show=True, plot=False)
        reduced_all, pca_all, scale_all = pca_analysis(timed_data, output_filename="",
                                                       plot_output_filename="",
                                                       columns=base_features+au_features+emotion_features, components=6,
                                                       color_by="NLG", show=True, plot=False)

    feature_dict = {"Base":reduced_base,
                    "wAUs":reduced_aus,
                    "wEmotions":reduced_emotions,
                    "All":reduced_all}

    last_time = output_time("Performing PCA Reduction", last_time)

    columns = get_columns(all_cols=list(data.columns), interval="All", type="Duration",
                          emotes=["Joy", "Frustration", "Confusion"])


    # print(data[response_column].describe())
    # print(sum(data[response_column] > 0))
    # print(data["PreTestScore"].describe())
    # print((data["Duration"]/60).describe())

    # poly_transform = PolynomialFeatures(degree=2, interaction_only=True)
    # kf = KFold(n_splits=10, shuffle=True)

    y = timed_data.loc[:, response_column]

    # regressors = {"Linear": LinearRegression()}
    # classifiers = {"LogReg": LogisticRegression(),
    #                "SVM.Lin":SVC(C=1.0, kernel='linear')}
    # prediction_type = "Classification"
    # model_keys = regressors.keys() if prediction_type == "Regression" else classifiers.keys()
    #
    # nlg_error_data = ErrorStruct(name=response_column, baseline_keyword="MeanBaseline", models=model_keys, keywords=feature_dict.keys())
    #
    # all_subjects = [e for e in list(data["TestSubject"].unique()) if e != "CI1301PN066"]
    #
    # # leave one subject out cross validation procedure
    # for subject in all_subjects:
    #     train = timed_data["TestSubject"] != subject
    #     test = timed_data["TestSubject"] == subject
    #     if prediction_type == "Regression":
    #         train_mean = y.loc[train].mean()
    #         mean_estimate_error = y.loc[test] - train_mean
    #         nlg_error_data.add_error_list(list(mean_estimate_error), model="MeanBaseline")
    #
    #         for reg_name in regressors.keys():
    #             for feature_set in feature_dict.keys():
    #                 if feature_set == "Base":
    #                     feature_data, pca, scale = pca_analysis(timed_data.loc[train, :],
    #                                                             columns=base_features,
    #                                                             components=6,
    #                                                             color_by="NLG",
    #                                                             show=False,
    #                                                             plot=False)
    #                 elif feature_set == "wEmotions":
    #                     feature_data, pca, scale = pca_analysis(timed_data.loc[train, :],
    #                                                               columns=base_features + emotion_features,
    #                                                               components=6, color_by="NLG",
    #                                                               show=False, plot=False)
    #                 elif feature_set == "wAUs":
    #                     feature_data, pca, scale = pca_analysis(timed_data.loc[train, :],
    #                                                             columns=base_features + au_features,
    #                                                             components=6, color_by="NLG", show=False, plot=False)
    #                 regressor = regressors[reg_name]
    #                 regressor.fit(X=feature_data.loc[train, :], y=y.loc[train])
    #
    #                 predictions = regressor.predict(X=feature_data.loc[test, :])
    #                 errors = y.loc[test] - predictions
    #                 nlg_error_data.add_error_list(list(errors), model=reg_name, keyword=feature_set)
    #     else:  # prediction_type == "Classification":
    #
    #         response = pd.Series(y < response_median, dtype=int)
    #         train_class = int(np.mean(response.loc[train]) + 0.5)
    #         mean_estimate_accuracy = np.array(np.equal(response.loc[test], train_class), dtype=int)
    #         nlg_error_data.add_error_list(list(mean_estimate_accuracy), model="MeanBaseline")
    #
    #         for clf_name in classifiers.keys():
    #             for feature_set in feature_dict.keys():
    #                 if feature_set == "Base":
    #                     feature_data, pca, scale = pca_analysis(timed_data.loc[train, :],
    #                                                             columns=base_features,
    #                                                             components=6,
    #                                                             color_by="NLG",
    #                                                             show=False,
    #                                                             plot=False)
    #                     feature_columns = base_features
    #                 elif feature_set == "wEmotions":
    #                     feature_data, pca, scale = pca_analysis(timed_data.loc[train, :],
    #                                                               columns=base_features + emotion_features,
    #                                                               components=6, color_by="NLG",
    #                                                               show=False, plot=False)
    #                     feature_columns = base_features + emotion_features
    #                 elif feature_set == "wAUs":
    #                     feature_data, pca, scale = pca_analysis(timed_data.loc[train, :],
    #                                                             columns=base_features + au_features,
    #                                                             components=6, color_by="NLG", show=False, plot=False)
    #                     feature_columns = base_features + au_features
    #                 elif feature_set == "All":
    #                     feature_data, pca, scale = pca_analysis(timed_data.loc[train, :],
    #                                                             columns=base_features + au_features + emotion_features,
    #                                                             components=6, color_by="NLG", show=False, plot=False)
    #                     feature_columns = base_features + au_features + emotion_features
    #                 else:
    #                     feature_data = pd.DataFrame()
    #                     pca = PCA()
    #                     scale = StandardScaler()
    #                     feature_columns = []
    #                 #feature_data = feature_dict[feature_set]
    #                 classifier = classifiers[clf_name]
    #                 classifier.fit(X=feature_data, y=response.loc[train])
    #
    #                 test_feature_data = apply_fitted_transformation(df=timed_data,
    #                                                                 columns=feature_columns,
    #                                                                 pca=pca,
    #                                                                 scaler=scale,
    #                                                                 standardize_by="GameTime")
    #                 predictions = classifier.predict(X=test_feature_data.loc[test, :])
    #                 accurates = np.array(np.equal(response.loc[test], predictions), dtype=int)
    #                 nlg_error_data.add_error_list(list(accurates), model=clf_name, keyword=feature_set)
    #
    #
    # last_time = output_time("Calculating Prediction Errors", last_time)
    # nlg_error_data.output_diagnostics()
    # baseline_errors, baseline_std, = nlg_error_data.error_by_interval(model="MeanBaseline")
    #
    # for model in model_keys:
    #     base_model_means, base_model_std = nlg_error_data.error_by_interval(model=model, keyword="Base")
    #     aus_model_means, aus_model_std = nlg_error_data.error_by_interval(model=model, keyword="wAUs")
    #     emotions_model_means, emotions_model_std = nlg_error_data.error_by_interval(model=model, keyword="wEmotions")
    #     all_model_means, all_model_std = nlg_error_data.error_by_interval(model=model, keyword="All")
    #
    #     # fig, ax = plt.subplots(1)
    #     # ax.set_title(model)
    #     # ax.set_xlabel("Time Interval")
    #     # ax.set_ylabel("Classification Accuracy")
    #     # ax.axhline(y=0.525424, color="red", label='Baseline')
    #     # ax.plot(list(np.arange(start=0, stop=9*5, step=5)), base_model_means.iloc[:9], color="blue", label="Base-SVM")
    #     # ax.plot(list(np.arange(start=0, stop=9*5, step=5)), emotions_model_means.iloc[:9], color="green", label="Emotions-SVM")
    #     # ax.legend(loc=4)
    #     # plt.show()
    #
    #     output_df = pd.concat([baseline_errors, base_model_means, aus_model_means, emotions_model_means, all_model_means], axis=1)
    #     output_df.columns = ["Baseline", "Base-%s" % model, "AUs-%s" % model, "Emotions-%s" % model, "All-%s" % model]
    #     print(output_df.iloc[:9, :])
    #     #print("Mean Difference: %.4f" % (output_df.loc[0:9, "Base-Linear"] - output_df.loc[0:9, "Emotions-Linear"]).mean())
    #     accuracy_df = pd.DataFrame()
    #
    #     baseline_last_mae, baseline_last_std_mae, baseline_last_mse, baseline_last_std_mse, baseline_all = \
    #         nlg_error_data.error_last_interval_per_fold(model="MeanBaseline")
    #     base_model_last_mae, base_model_last_std_mae, base_model_last_mse, base_model_last_std_mse, base_model_all = \
    #         nlg_error_data.error_last_interval_per_fold(model=model, keyword="Base")
    #     aus_model_last_mae, aus_model_last_std_mae, aus_model_last_mse, aus_model_last_std_mse, aus_model_all = \
    #         nlg_error_data.error_last_interval_per_fold(model=model, keyword="wAUs")
    #     emotions_model_last_mae, emotions_model_last_std_mae, emotions_model_last_mse, emotions_model_last_std_mse, emotions_model_all = \
    #         nlg_error_data.error_last_interval_per_fold(model=model, keyword="wEmotions")
    #     all_model_last_mae, all_model_last_std_mae, all_model_last_mse, all_model_last_std_mse, all_model_all = \
    #         nlg_error_data.error_last_interval_per_fold(model=model, keyword="All")
    #     accuracy_df["MSE"] = pd.Series([baseline_last_mse, base_model_last_mse, aus_model_last_mse, emotions_model_last_mse, all_model_last_mse],
    #                     index=["Baseline", "Base-%s" % model, "AUs-%s" % model, "Emotions-%s" % model, "All-%s" % model])
    #     accuracy_df["MAE"] = pd.Series([baseline_last_mae, base_model_last_mae, aus_model_last_mae, emotions_model_last_mae, all_model_last_mae],
    #                     index=["Baseline", "Base-%s" % model, "AUs-%s" % model, "Emotions-%s" % model, "All-%s" % model])
    #     accuracy_df["MSE-Std"] = pd.Series([baseline_last_std_mse, base_model_last_std_mse, aus_model_last_std_mse, emotions_model_last_std_mse, all_model_last_std_mse],
    #                     index=["Baseline", "Base-%s" % model, "AUs-%s" % model, "Emotions-%s" % model, "All-%s" % model])
    #     accuracy_df["MAE-Std"] = pd.Series([baseline_last_std_mae, base_model_last_std_mae, aus_model_last_std_mae, emotions_model_last_std_mae, all_model_last_std_mae],
    #                     index=["Baseline", "Base-%s" % model, "AUs-%s" % model, "Emotions-%s" % model, "All-%s" % model])
    #     accuracy_df["All-MAE"] = pd.Series([baseline_all, base_model_all, aus_model_all, emotions_model_all, all_model_all],
    #                     index=["Baseline", "Base-%s" % model, "AUs-%s" % model, "Emotions-%s" % model, "All-%s" % model])
    #
    #     print(accuracy_df)
    #
    # print(final_data.loc[:,base_features].apply(lambda col: (col / (final_data.loc[:, "GameTime"] / 60.0))).mean())
    # print(final_data.loc[:,"PreTestScore"].describe())
    # log_reg_final_output(data=final_data, features=base_features, pca_output=output_filenames["BasePCA"],
    #                      plot_output=output_filenames["BasePCAPlot"], response=response_column)
    log_reg_final_output(data=final_data, features=base_features+emotion_features, pca_output=output_filenames["EmotionPCA"],
                         plot_output=output_filenames["EmotionPCAPlot"], response=response_column, a_output=output_filenames["SVMPCA_Transform"])
    # log_reg_final_output(data=final_data, features=base_features+au_features, pca_output=output_filenames["AUPCA"],
    #                      plot_output=output_filenames["AUPCAPlot"], response=response_column)
    # log_reg_final_output(data=final_data, features=base_features+emotion_features+au_features, pca_output=output_filenames["AllPCA"],
    #                      plot_output=output_filenames["AllPCAPlot"], response=response_column)

    plot_SVM_boundary(data=final_data, features=base_features+emotion_features, pca_output=output_filenames["EmotionPCA"],
                      plot_output=output_filenames["EmotionPCAPlot"], boundary_output=output_filenames["SVMBoundary"],
                      f1=0, f2=1, response=response_column)

    # # print(np.array(errors_dict["Ridge"]).shape)
    # print("----------------------------------------------------------------")
    # for reg_name in errors_dict.keys():
    #     print("Accuracy predicting %d in LOOCV using %s with %d features" %
    #           (len(errors_dict[reg_name]), reg_name, data_X.shape[1]))
    #     print("MAE: %.4f" % (np.mean(np.abs(np.array(errors_dict[reg_name])))))
    #     print("MSE: %.4f" % (np.mean(np.array(errors_dict[reg_name])**2)))
    #     print("----------------------------------------------------------------")

