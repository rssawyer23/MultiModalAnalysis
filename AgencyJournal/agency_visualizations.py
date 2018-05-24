import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import AgencyJournal.minor_functions as mf


def add_agency_legend(ax, loc):
    full = mpatches.Patch(color=tuple([0.0, 1.0, 0.0]), label="High Agency")
    partial = mpatches.Patch(color=tuple([0.0, 0.0, 1.0]), label="Low Agency")
    none = mpatches.Patch(color=tuple([1.0, 0.0, 0.0]), label="No Agency")
    legend_handles = [full, partial, none]
    ax.legend(handles=legend_handles, loc=loc)


def plot_scatter(df, x_values, y_values, color_by="Condition", save_filepath="C:/Users/robsc/Documents/NC State/GRAWork/Publications/AgencyJournal/Images"):
    """Plot a scatter plot of all (x_values) by (y_values) using (color_by) to color vertices by a grouping color"""
    fig, ax = plt.subplots(1)
    ax.set_xlabel(x_values)
    ax.set_ylabel(y_values)
    ax.set_title("%s versus %s colored by %s" % (x_values, y_values, color_by))
    colors = df.loc[:, color_by].apply(mf.color_map)
    ax.scatter(df.loc[:, x_values], df.loc[:, y_values], c=colors)

    add_agency_legend(ax, loc=4)

    fig.savefig(save_filepath)
    plt.close()


def plot_condition_histogram(df, x_values, condition, save_filepath):
    """Plot histogram of (x_values) for (condition) using seaborn"""
    fig, ax = plt.subplots(1)
    condition_rows = df.loc[:, "Condition"] == condition
    ax = sns.distplot(df.loc[condition_rows, x_values],
                      hist=True, kde=True, rug=True,
                      color=mf.color_map(condition))
    ax.set_xlabel(x_values)
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of %d Condition's %s" % (condition, x_values))
    fig.savefig(save_filepath)
    plt.close()


def plot_all_conditions_histogram(df, x_values, save_filepath):
    """Plot histograms of (x_values) for each of the three conditions on single figure"""
    fig, ax = plt.subplots(1, 3)
    for i in range(3):
        condition = i + 1
        condition_rows = df.loc[:, "Condition"] == condition
        sns.distplot(df.loc[condition_rows, x_values],
                     hist=True, kde=True, rug=True,
                     color=mf.color_map(condition),
                     ax=ax[i])
        ax[i].set_title("Condition %d" % condition)
    fig.savefig(save_filepath)
    plt.close()


def plot_all_conditions_density(df, x_values, save_filepath):
    """Plot density curve estimates for each condition on the same plot"""
    fig, ax = plt.subplots(1)
    for condition in range(1, 4):
        condition_rows = df.loc[:, "Condition"] == condition
        sns.distplot(df.loc[condition_rows, x_values],
                     hist=False, kde=True, rug=True,
                     color=mf.color_map(condition),
                     ax=ax)
    ax.set_ylabel("Density")
    ax.set_xlabel("Normalized Learning Gain")
    add_agency_legend(ax, loc=1)
    fig.savefig(save_filepath)
    plt.close()


def plot_bar_graph_duration_intervals(act_df, save_filepath):
    new_df = pd.DataFrame(index=["TestSubject", "Interval", "Duration", "Condition"])
    for i, row in act_df.iterrows():
        cond = mf.condition_map(row["Condition"])
        tutorial_row = pd.Series(data=[row["TestSubject"], "Tutorial", row["Duration-Tutorial"] / 60.0, cond],
                                 index=["TestSubject", "Interval", "Duration", "Condition"])
        diagnosis_row = pd.Series(data=[row["TestSubject"], "Diagnosis", row["Duration-PreScan"] / 60.0, cond],
                                  index=["TestSubject", "Interval", "Duration", "Condition"])
        testing_row = pd.Series(
            data=[row["TestSubject"], "Hypothesis Testing", row["Duration-PostScan"] / 60.0, cond],
            index=["TestSubject", "Interval", "Duration", "Condition"])
        new_df = pd.concat([new_df, tutorial_row, diagnosis_row, testing_row], axis=1)
    new_df = new_df.T
    new_df.index = list(range(new_df.shape[0]))

    fig, ax = plt.subplots(1)
    sns.barplot(x="Interval", y="Duration", hue="Condition", data=new_df, ax=ax, palette=mf.COND_COLOR_MAP)
    ax.set_ylabel("Duration (min)")
    fig.savefig(save_filepath)
    plt.close()
    return new_df
