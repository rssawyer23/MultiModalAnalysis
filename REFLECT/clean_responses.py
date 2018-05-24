import pandas as pd

VGPF_MAP = {"Not at all":1,
            "Rarely":2,
            "Occasionally":3,
            "Frequently":4,
            "Very Frequently":5}
VGPS_MAP = {"Not at all skilled":1,
            "Limited skills":2,
            "Average":3,
            "Skilled":4,
            "Very skilled":5}
VGPH_MAP = {"0-2 hours":1,
            "3-5 hours":2,
            "5-10 hours":3,
            "10-20 hours":4,
            "Over 20 hours":5}


def map_cell(cell, map_dict):
    if cell in map_dict.keys():
        return map_dict[cell]
    elif cell in map_dict.values():
        return cell
    else:
        print("ERROR MAPPING %s" % cell)
        return cell


def clean_responses_to_numeric(activity_summary_inpath, activity_summary_outpath):
    data = pd.read_csv(activity_summary_inpath)

    data["VideoGamePlayingFrequency"] = data["VideoGamePlayingFrequency"].apply(map_cell, map_dict=VGPF_MAP)
    data["VideoGamePlayingSkill"] = data["VideoGamePlayingSkill"].apply(map_cell, map_dict=VGPS_MAP)
    data["VideoGamePlayingHoursPerWeek"] = data["VideoGamePlayingHoursPerWeek"].apply(map_cell, map_dict=VGPH_MAP)

    data.to_csv(activity_summary_outpath, index=False)


def clean_student_rows(appended_activity_summary_filename, edited_activity_summary_filename):
    focus_group = ['1A0011', '1A0019', '1B0038', '2A0081', '2A0083', '2B0106', '3A0127', '3B0153']
    one_class = ["1B0053", "1B0048", "3B0154", "1B0054", "1B0043", "2A0063"]
    no_post = ['1B0043', '1B0048', '1B0051', '1B0053', '1B0054', '1B0055', '2A0063', '2A0067', '2A0075', '3B0148',
               '3B0156', '3B0158', '3B0167']
    removal_set = set(focus_group) | set(one_class) | set(no_post)
    act_df = pd.read_csv(appended_activity_summary_filename)
    keep_rows = act_df["TestSubject"].apply(lambda cell: cell not in removal_set)
    act_df = act_df.loc[keep_rows, :]
    act_df.to_csv(edited_activity_summary_filename, index=False)

if __name__ == "__main__":
    in_filepath = "C:/Users/robsc/Documents/NC State/GRAWork/CI-REFLECT/REFLECT-2-2018/Output/ActivitySummary/ActivitySummaryGradedEdited.csv"
    out_filepath = in_filepath  # Overwriting the input filepath
    clean_responses_to_numeric(activity_summary_inpath=in_filepath,
                               activity_summary_outpath=out_filepath)
