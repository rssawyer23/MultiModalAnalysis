import pandas as pd
import datetime
from dateutil.parser import parse

if __name__ == "__main__":
    gp_fp = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/EventSequence/GoldenPathEventSequence_wC.csv"
    ev_fp = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/EventSequence/EventSequence_wC.csv"
    out_gp = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/EventSequence/GoldenPathEventSequenceTimeStamp.csv"
    dummy_start = parse(pd.read_csv(ev_fp)["TimeStamp"].iloc[0])
    d = pd.read_csv(gp_fp)
    d["TimeStamp"] = d["GameTime"].apply(lambda cell: dummy_start + datetime.timedelta(seconds=float(cell)))
    d.to_csv(out_gp, index=False)
