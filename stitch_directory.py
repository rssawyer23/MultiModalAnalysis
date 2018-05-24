import pandas as pd
import os


def stitch_directory(dir, out_name=""):
    file_list = os.listdir(directory)
    df = pd.DataFrame()
    for file_name in file_list:
        a = pd.read_csv(directory + "/" + file_name)
        df = pd.concat([df, a])
    if out_name == "":
        out_name = dir.split("/")[-1] + ".csv"

    df.to_csv(dir + "/" + out_name, index=False)


def stitch_similar_files(fn_one, fn_two, out_fn):
    df_one = pd.read_csv(fn_one)
    df_two = pd.read_csv(fn_two)
    assert df_one.shape[1] == df_two.shape[1]
    df_merged = pd.concat([df_one, df_two])
    df_merged.to_csv(out_fn, index=False)

#
# directory = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/TDP/Output/FACET-Z-Score"
# stitch_directory(directory)

full_dir = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/OutputFull2018/FACET-ThresholdCrossed/"
facet_ab = full_dir + "FACET-Events.csv"
facet_c = full_dir + "FACET-Events-NoAgency.csv"
out_abc = full_dir + "FACET-Events-All.csv"
stitch_similar_files(facet_ab, facet_c, out_abc)
