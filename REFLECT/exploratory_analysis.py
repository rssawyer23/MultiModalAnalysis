import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


def quick_tests(df, group_rows, response):
    print(df[response].describe())
    print("In Group:%d, Out Group:%d" % (sum(group_rows), sum(~group_rows)))
    sns.distplot(df.loc[group_rows, response], rug=True)
    sns.distplot(df.loc[~group_rows, response], rug=True)
    t, p = ttest_ind(df.loc[group_rows, response], df.loc[~group_rows, response])
    print("t-value:%.4f, p-val:%.4f" % (t,p))


focus_group = ['1A0011', '1A0019', '1B0038', '2A0081', '2A0083', '2B0106', '3A0127', '3B0153']
one_class = ["1B0053", "1B0048", "3B0154", "1B0054", "1B0043", "2A0063"]
no_post = ['1B0043', '1B0048', '1B0051', '1B0053', '1B0054', '1B0055', '2A0063', '2A0067', '2A0075', '3B0148', '3B0156', '3B0158', '3B0167']
removal_set = set(focus_group) | set(one_class) | set(no_post)

reflect_data_directory = "C:/Users/robsc/Documents/NC State/GRAWork/CI-REFLECT/REFLECT-2-2018/Output/"
act_sum = reflect_data_directory + "ActivitySummary/ActivitySummaryGraded.csv"

df = pd.read_csv(act_sum)
keep_rows = df["TestSubject"].apply(lambda cell: cell not in removal_set)
df = df.loc[keep_rows, :]
df.index = list(range(df.shape[0]))

complete_rows = df["MysterySolved"]
female_rows = df["Gender"] == "Female"
flu_rows = df["SolutionDisease"] == "Influenza"

quick_tests(df, complete_rows, "Duration")
quick_tests(df, complete_rows, "Post-IMI")
quick_tests(df, complete_rows, "Post-IMI-Interest-Enjoyment")
quick_tests(df, flu_rows, "Duration")
quick_tests(df, flu_rows, "Post-IMI")
quick_tests(df, female_rows, "Post-IMI")


fig, ax = plt.subplots(1)
sns.distplot(df.loc[complete_rows, "Duration"], hist=False, rug=True, color=tuple([0.0, 1.0, 0.0]), ax=ax)
sns.distplot(df.loc[~complete_rows, "Duration"], hist=False, rug=True, color=tuple([1.0, 0.0, 0.0]), ax=ax)

leads_data_directory = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output318/"
leads_act_sum = leads_data_directory + "ActivitySummary/ActivitySummaryAppendedRevisedEdited.csv"
leads_df = pd.read_csv(leads_act_sum)

leads_complete_rows = leads_df["MysterySolved"]
leads_flu_rows = leads_df["SolutionDisease"] == "Influenza"
leads_female_rows = leads_df["Gender"] == 2

quick_tests(leads_df, leads_complete_rows, "Duration")
quick_tests(leads_df, leads_complete_rows, "Post-IMI")
quick_tests(leads_df, leads_flu_rows, "Post-IMI")
quick_tests(leads_df, leads_female_rows, "Post-IMI")
