from datetime import datetime, timezone
from dateutil import parser
from facet_event_file import index_dictionary
# Script for merging the FACET-Events.csv file with the EventSequence.csv file
# facet_filename should be FACET-Events.csv file, which is unchanged
# event_filename should be EventSequence.csv file, which is unchanged (and is only used for the header)
# output_filename is the EventSequence file that now includes the FACET events in the proper chronological location

facet_filename = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/FACET-ThresholdCrossed/FACET-Events.csv"
output_filename = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/EventSequence/EventSequenceFACET.csv"
event_filename = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/EventSequence/EventSequence.csv"


def get_line(previous_line, previous_time, previously_written, previous_subject, rfile, ind):
    if previously_written:  # This means the line from the file was written and need to get the next line
        line = rfile.readline()
        line_split = line.replace("\n", "").replace("\r", "").split(",")
        if len(line_split) > 1:
            line_time = parser.parse(line_split[ind["TimeStamp"]])
            line_subject = line_split[ind["TestSubject"]]
        else:  # This is what the variables are set to once a file has reached the end
            check_line = rfile.readline()  # Ensuring there are two blank lines in a row (since getting eof too early)
            line_split = check_line.replace("\n", "").replace("\r", "").split(",")
            if len(line_split) > 1:
                line_time = parser.parse(line_split[ind["TimeStamp"]])
                line_subject = line_split[ind["TestSubject"]]
            else:
                line_time = datetime.now(timezone.utc)  # Should guarantee time stamp comparison never has earlier
                line_subject = "999999999"  # Should guarantee test subject comparison never has earlier
    else:  # This means the line from the file was not written and should use the previous line until written
        line = previous_line
        line_time = previous_time
        line_subject = previous_subject
    line_active = line_subject != '999999999'
    return line, line_time, line_active, line_subject


# This is the primary function (main function)
def merge_event_files(event_filename, facet_filename, event_facet_filename, show=False):
    with open(event_filename, 'r') as efile, open(facet_filename, 'r') as ffile, open(event_facet_filename, 'w') as ofile:
        start_time = datetime.now()
        header = efile.readline()
        eind = index_dictionary(header, [])
        find = index_dictionary(ffile.readline(), [])

        ofile.write(header)

        # Initialization, not actually used
        event_line = header
        event_time = header.replace("\n", "").replace("\r", "").split(",")[eind["TimeStamp"]]
        event_subject = header.replace("\n", "").replace("\r", "").split(",")[eind["TestSubject"]]
        facet_line = header
        facet_time = header.replace("\n", "").replace("\r", "").split(",")[find["TimeStamp"]]
        facet_subject = header.replace("\n", "").replace("\r", "").split(",")[eind["TestSubject"]]

        # Initialization, actually used
        event_active = True
        event_written = True
        facet_active = True
        facet_written = True
        line_count = 0
        event_count = 0
        facet_count = 0

        while event_active or facet_active:
            event_line, event_time, event_active, event_subject = get_line(event_line, event_time, event_written, event_subject, efile, eind)
            facet_line, facet_time, facet_active, facet_subject = get_line(facet_line, facet_time, facet_written, facet_subject, ffile, find)
            # Checking which file is "ahead" by comparing timestamps and test subject number, equal test subject number is ok
            try:
                if float(event_subject[5]+event_subject[-3:]) > float(facet_subject[5]+facet_subject[-3:]) and facet_active:
                    ofile.write(facet_line)
                    event_written = False
                    facet_written = True
                    facet_count += 1
                elif float(event_subject[5]+event_subject[-3:]) < float(facet_subject[5]+facet_subject[-3:]) and event_active:
                    ofile.write(event_line)
                    event_written = True
                    facet_written = False
                    event_count += 1
                elif event_time >= facet_time and facet_active:
                    ofile.write(facet_line)
                    event_written = False
                    facet_written = True
                    facet_count += 1
                elif event_time < facet_time and event_active:
                    ofile.write(event_line)
                    event_written = True
                    facet_written = False
                    event_count += 1

                line_count += 1

                # Line counts to see progress
                if event_count % 10000 == 1 and show:
                    print("Event Lines: %d" % (event_count-1))
                if facet_count % 10000 == 1 and show:
                    print("FACET Lines: %d" % (facet_count-1))
                if line_count % 10000 == 1 and show:
                    print("Total Lines: %d" % (line_count-1))
            except TypeError:
                print("ESub: %s, FSub: %s, ETS: %s, FTS: %s" % (event_subject, facet_subject, event_time, facet_time))
        minutes_elapsed = (datetime.now() - start_time).total_seconds() / 60.
        print("Successfully merged EventSequence and FACET-Events into %s in %.3f minutes" % (output_filename, minutes_elapsed))

if __name__ == "__main__":
    merge_event_files(event_filename, facet_filename, output_filename, show=True)
