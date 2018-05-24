import pandas as pd
import seaborn as sns


CONDITION_MAP = {1:"High Agency",
                 2:"Low Agency",
                 3:"No Agency"}

COLOR_MAP = {1:tuple([0.0, 1.0, 0.0]),
             2:tuple([0.0, 0.0, 1.0]),
             3:tuple([1.0, 0.0, 0.0])}

COND_COLOR_MAP = {"High Agency":tuple([0.2, 1.0, 0.2]),
                  "Low Agency":tuple([0.2, 0.2, 1.0]),
                  "No Agency":tuple([1.0, 0.2, 0.2])}


def condition_map(x):
    if x in CONDITION_MAP.keys():
        return CONDITION_MAP[x]
    else:
        print("ERROR: unmatched condition: %d" % x)


def color_map(x):
    if x in COLOR_MAP.keys():
        return COLOR_MAP[x]
    else:
        print("ERROR: unmatched color_by %s" % x)
        return tuple([0.0, 0.0, 0.0])