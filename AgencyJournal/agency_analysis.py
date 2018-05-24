import pandas as pd
import AgencyJournal.manual_statistics as ms
import AgencyJournal.agency_visualizations as av
from AgencyJournal.minor_functions import condition_map

NO_SURVEY = ["CI1302PN011"]
NO_TRACE = ["CI1301PN035", "CI1301PN037", "CI1301PN042", "CI1301PN043", "CI1301PN073"]


def print_demographics(df, output_columns, title_key):
    total_students = df.shape[0]
    females = sum(df["Gender"] == 2) if str(df.loc[0, "Gender"]).isdigit() else sum(df["Gender"] == "Female")
    finished = sum(df["MysterySolved"])
    pos_lg = sum(df["RevisedLG"] > 0) if "RevisedLG" in df.columns else sum(df["LearningGain"] > 0)
    zero_lg = sum(df["RevisedLG"] == 0) if "RevisedLG" in df.columns else sum(df["LearningGain"] == 0)
    warning = "WARNING: " if title_key == "BOTH" else ""
    print("-----------------------------------------------------------------------------------------------")
    print("%s Demographic Data" % title_key)
    print("\t%d Students Total" % total_students)
    print("\t%s%d Females (%.4f)" % (warning, females, females/total_students))
    print("\t%d solved the mystery (%.4f)" % (finished, finished/total_students))
    print("\t%s%d had positive learning gains (%.4f), %d had no learning gains (%.4f)" %
          (warning, pos_lg, pos_lg/total_students, zero_lg, zero_lg/total_students))
    print(df.loc[:, output_columns].describe().loc[["mean", "std", "min", "max"]].T)
    print("-----------------------------------------------------------------------------------------------")


def print_condition_table(df, output_columns):
    return_df = pd.DataFrame()
    df["PositiveLG"] = pd.Series(df["RevisedLG"] > 0, dtype=int)

    for index in [0, 1, 2, 3]:
        if index != 0:
            cond_rows = df["Condition"] == index
        else:
            cond_rows = pd.Series([True] * df.shape[0])
        m_series = df.loc[cond_rows, output_columns].mean()
        m_series.index = ["%s-Mean" % e for e in m_series.index]
        s_series = df.loc[cond_rows, output_columns].std()
        s_series.index = ["%s-Std" % e for e in s_series.index]
        set_size = pd.Series(data=df.loc[cond_rows, :].shape[0], index=["Participants"])
        full_series = pd.concat([m_series, s_series, set_size])
        return_df = pd.concat([return_df, full_series], axis=1)
    return_df = return_df.T
    return_df.index = ["All", "High", "Low", "No"]
    print(return_df)
    return return_df


def print_action_comparison_table(df, output_columns):
    return_df = pd.DataFrame()

    for col in output_columns:
        high_rows = df["Condition"] == 1
        low_rows = df["Condition"] == 2
        high_mean, high_std = df.loc[high_rows, col].mean(), df.loc[high_rows, col].std()
        low_mean, low_std = df.loc[low_rows, col].mean(), df.loc[low_rows, col].std()

        t_series = ms.welchs_ttest(df.loc[high_rows, col], df.loc[low_rows, col])
        t = t_series.loc["t-stat"]
        p_val = t_series.loc["p-value"]
        eff = t_series.loc["Effect"]

        full_series = pd.Series(data=[high_mean, high_std, low_mean, low_std, t, p_val, eff],
                                index=["High Mean", "High Std", "Low Mean", "Low Std", "t-stat", "p-value", "effect"])
        return_df = pd.concat([return_df, full_series], axis=1)
    return_df = return_df.T
    return_df.index = output_columns
    return_df.sort_values(by="p-value", axis=0, inplace=True, ascending=True)
    return_df["Holm-Bon"] = ms.add_holm_bonferroni(return_df.loc[:, "p-value"], return_df.index)
    print(return_df.round(2))
    return return_df


def remove_excess(df, subject_omit_list):
    """Remove subjects from subject omit list (keep students not in the list)"""
    keep_rows = df["TestSubject"].apply(lambda x: x not in subject_omit_list)
    df = df.loc[keep_rows, :]
    df.index = list(range(df.shape[0]))
    return df


def generate_duration_table(act_df, show='none'):
    new_df = pd.DataFrame(index=["", "-Tutorial", "-PreScan", "-PostScan"])
    for cond in [1, 2, 3]:
        cond_rows = act_df["Condition"] == cond
        cond_m_series = pd.Series(index=["", "-Tutorial", "-PreScan", "-PostScan"])
        cond_s_series = pd.Series(index=["", "-Tutorial", "-PreScan", "-PostScan"])
        for interval in ["", "-Tutorial", "-PreScan", "-PostScan"]:
            cond_m_series[interval] = (act_df.loc[cond_rows, "Duration%s" % interval] / 60.0).mean()
            cond_s_series[interval] = (act_df.loc[cond_rows, "Duration%s" % interval] / 60.0).std()
        new_df[condition_map(cond)+"-Mean"] = cond_m_series
        new_df[condition_map(cond)+"-Std"] = cond_s_series
    if show == "maximum":
        print(new_df)
    return new_df


def basic_visualizations():
    """Plotting basic visualizations of each condition for specific variables of interest"""
    av.plot_scatter(act_df, x_values="TotalDuration", y_values="RevisedNLG", color_by="Condition",
                    save_filepath=image_output_stem + "/DurationNLGScatter.png")
    av.plot_condition_histogram(act_df, x_values="RevisedNLG", condition=1,
                                save_filepath=image_output_stem + "/FullAgencyNLGHist.png")
    av.plot_condition_histogram(act_df, x_values="RevisedNLG", condition=2,
                                save_filepath=image_output_stem + "/PartialAgencyNLGHist.png")
    av.plot_condition_histogram(act_df, x_values="RevisedNLG", condition=3,
                                save_filepath=image_output_stem + "/NoAgencyNLGHist.png")
    av.plot_all_conditions_histogram(act_df, x_values="RevisedNLG",
                                     save_filepath=image_output_stem + "/ConditionsNLGHist.png")
    av.plot_all_conditions_density(act_df, x_values="RevisedNLG",
                                   save_filepath=image_output_stem + "/ConditionsDensities.png")
    av.plot_bar_graph_duration_intervals(act_df, save_filepath=image_output_stem + "/IntervalDurations.png")

if __name__ == "__main__":
    activity_summary_filename = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/OutputFull2018/ActivitySummary/ActivitySummaryAppendedRevisedEdited-All.csv"
    image_output_stem = "C:/Users/robsc/Documents/NC State/GRAWork/Publications/AgencyJournal/Images"

    act_df = pd.read_csv(activity_summary_filename)
    act_df["TotalDuration"] = act_df["Duration-Tutorial"] + act_df["Duration-PreScan"] + act_df["Duration-PostScan"]
    act_df["NLG per Time"] = act_df["RevisedNLG"] / (act_df["TotalDuration"] / 60.0)

    # ANOVAs
    # ms.anova(act_df["Condition"], act_df["TotalDuration"] / 60.0, show="maximum")
    #ms.anova(act_df["Condition"], act_df["RevisedNLG"], show='maximum')
    ms.posthoc(act_df["Condition"], act_df["RevisedNLG"], show='maximum')

    # generate_duration_table(act_df, show='maximum')

    act_df = remove_excess(act_df, subject_omit_list=NO_SURVEY)
    # print(act_df["RevisedNLG"].describe())

    # Some basic plots including histograms/densities of condition learning
    # basic_visualizations()

    # Demographics for Section 3: Experimental Methodology
    print_demographics(df=act_df, output_columns=["RevisedNLG", "TotalDuration", "Age", "Condition"],
                       title_key="FULL LEADS")

    print_condition_table(df=act_df, output_columns=["RevisedPre", "RevisedPost", "RevisedNLG", "PositiveLG"])

    # For Section 4.2: Impact of agency on learning
    act_df.to_csv("C:/Users/robsc/Documents/GitHub/MultiModalAnalysis/AgencyJournal/ActivitySummaryR.csv", index=False)

    # For Section 4.3: Impact of agency on problem-solving behaviors
    act_df = remove_excess(act_df, subject_omit_list=NO_TRACE)
    print_action_comparison_table(df=act_df, output_columns=["PostScan-BooksAndArticles-Count",
                                                             "PostScan-Conversation-Count",
                                                             "PostScan-Scanner-Count",
                                                             "PostScan-Worksheet-Count",
                                                             "PostScan-WorksheetSubmit-Count"])
    act_df.to_csv("C:/Users/robsc/Documents/GitHub/MultiModalAnalysis/AgencyJournal/ActivitySummaryR_NoTrace.csv", index=False)
