from datetime import datetime
from dateutil.parser import parse
# Script for converting the FACET-Threshold-Crossed.csv file into an EventSequence-like file
# input_filename should be FACET-ThresholdCrossed.csv file, which is unchanged
# event_filename should be EventSequence.csv file, which is unchanged (and is only used for the header)
# output_filename is the EventSequence-like file of FACET-ThresholdCrossed.csv file

input_filename = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/FACET-ThresholdCrossed/FACET-ThresholdCrossed.csv"
only_positives = True
duration_min = 0.5
output_filename = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/FACET-ThresholdCrossed/FACET-Events.csv"
event_filename = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/EventSequence/EventSequence.csv"


# To generalize headers instead of using specific indices
def index_dictionary(header_line, extras):
    in_dict = dict()
    header_split = header_line.replace("\n","").replace("\r","").split(",")
    for ele, index in zip(header_split,range(len(header_split))):
        in_dict[ele] = index
    for e, i in zip(extras, range(len(header_split), len(header_split) + len(extras))):
        in_dict[e] = i
    return in_dict


# Go through headers of the EventSequence and write values of line from FACET-ThresholdCrossed that match
def map_line(eheader, ls, ind):
    o_list = []
    for e in eheader.split(","):
        try:
            if e == "Event":  # Making the Event field be FACET
                o_list.append("FACET")
            elif e == "Target":  # Making the Target field be the Channel of FACET (specific AU or Emotion)
                o_list.append(ls[ind["Channel"]])
            else:
                o_list.append(ls[ind[e]])
        except KeyError:
            o_list.append("")
    return ",".join(o_list) + "\n"


# This is the primary function (main function)
def create_facet_event_file(facet_thresh_filename, facet_event_filename, event_filename, only_positives=True, duration_min=0.5, absolute_min=0.5):
    with open(facet_thresh_filename, 'r') as rfile, open(facet_event_filename, 'w') as ofile:
        start_time = datetime.now()
        print("Started FACET-Event file %s" % start_time)
        header = rfile.readline()
        ind = index_dictionary(header, [])
        output_buffer = []

        # Stealing header from event file these files should match
        with open(event_filename, 'r') as efile:
            eheader = efile.readline()
            ofile.write(eheader)

        prev_test_subject = ""
        facet_starts = dict()
        for line in rfile:
            ls = line.split(",")
            test_subject = ls[ind["TestSubject"]]
            if test_subject != prev_test_subject:
                if prev_test_subject != "":
                    output_buffer = sorted(output_buffer, key=lambda tup: tup[0])
                    for _, oline in output_buffer:
                        ofile.write(oline)
                output_buffer = []
                facet_starts = dict()

            # Verifying a valid row
            if len(ls) > 7:
                # Start tracking FACET event if
                #       above threshold and
                #       is positive and
                #       true evidence value above an absolute evidence score minimum
                if ls[ind["Target"]] == "BeginAboveThreshold":
                    if ls[ind["Direction"]] == "Positive" and float(ls[ind["Value"]]) > absolute_min:
                        facet_starts[ls[ind["Channel"]]] = ls[ind["TimeStamp"]]
                    elif only_positives:  # Not a positive event and do not want to keep
                        facet_starts[ls[ind["Channel"]]] = "Invalid"
                    else:
                        facet_starts[ls[ind["Channel"]]] = ls[ind["TimeStamp"]]
                # When an EndAbove is found
                #   use FACET dictionary to add the duration of the event to the corresponding start
                elif ls[ind["Target"]] == "EndAboveThreshold":
                    duration = float(ls[ind["Duration"]])
                    try:
                        start_timestamp = facet_starts[ls[ind["Channel"]]]
                    except KeyError:
                        start_timestamp = "Invalid"
                    if start_timestamp != "Invalid" and duration > duration_min:
                        ls[ind["TimeStamp"]] = start_timestamp
                        output_line = map_line(eheader, ls, ind)
                        output_buffer.append((parse(start_timestamp), output_line))

            prev_test_subject = test_subject
        minutes_elapsed = (datetime.now() - start_time).total_seconds() / 60.
        print("Successfully created %s in %.3f minutes" % (output_filename, minutes_elapsed))

if __name__ == "__main__":
    create_facet_event_file(input_filename, output_filename, event_filename, only_positives, duration_min)