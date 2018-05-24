from facet_event_file import index_dictionary
from datetime import datetime
from dateutil import parser



def parse_event_into_intervals(event_facet_filename):
    intervals = ["Tutorial.csv", "PreScan.csv", "PostScan.csv"]
    start_time_function = datetime.now()

    # Open three separate files for each of the intervals
    with open(event_facet_filename, 'r') as rfile, open(event_facet_filename[:-4]+intervals[0], 'w') as ofile0, open(event_facet_filename[:-4]+intervals[1], 'w') as ofile1, open(event_facet_filename[:-4]+intervals[2], 'w') as ofile2:
        header = rfile.readline()
        ind = index_dictionary(header, [])
        ofile0.write(header)
        ofile1.write(header)
        ofile2.write(header)

        # Initialize looping parameters
        prev_test_subject = ""
        passed_tutorial = False
        passed_scan = False
        no_agency_subject = False

        # Loop through the merged event_facet file to write lines based on which interval they are in
        for line in rfile:
            line_split = line.replace("\n","").replace("\r","").split(",")
            if len(line_split) > 5:
                current_test_subject = line_split[ind["TestSubject"]]
                # If subject has changed, reset the interval indicator booleans (and start time for no agency)
                if prev_test_subject != current_test_subject:
                    no_agency_subject = current_test_subject[5] == "3"
                    passed_tutorial = False
                    passed_scan = False
                    start_time = parser.parse(line_split[ind["TimeStamp"]]) # Parsed datetime for subtracting with others

                # If the subject has not finished the tutorial, write the line to the tutorial file
                if not passed_tutorial:
                    ofile0.write(line)
                    # Check if this line is the indicator of finishing the tutorial
                    if not no_agency_subject and line_split[ind["Location"]] != "" and line_split[ind["Location"]] != "Dock" and "Beach" not in line_split[ind["Location"]]:
                        passed_tutorial = True
                    if no_agency_subject and (parser.parse(line_split[ind["TimeStamp"]]) - start_time).total_seconds() > 402:
                        passed_tutorial = True
                # If the subject has not scanned an item, write the line to the PreScan interval file
                elif not passed_scan:
                    ofile1.write(line)
                    # Check if this line is the indicator of scanning an item
                    if not no_agency_subject and line_split[ind["Event"]] == "PlotPoint" and line_split[ind["Name"]] == "TestObject":
                        passed_scan = True
                    if no_agency_subject and (parser.parse(line_split[ind["TimeStamp"]]) - start_time).total_seconds() > 4628 + 402:
                        passed_scan = True
                # If finished the tutorial and scanning, write the line to the PostScan interval file
                elif passed_tutorial and passed_scan:
                    ofile2.write(line)

                prev_test_subject = current_test_subject
    minutes_elapsed = (datetime.now() - start_time_function).total_seconds() / 60.
    print("Successfully split EventSequence into the following files in %.3f minutes:\n\t%s\n\t%s\n\t%s" %
          (minutes_elapsed, event_facet_filename[:-4]+intervals[0], event_facet_filename[:-4]+intervals[1],
           event_facet_filename[:-4] + intervals[2]))

if __name__ == "__main__":
    # input_filename = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/OutputFull2018/EventSequence/EventSequenceFACET.csv"
    # parse_event_into_intervals(event_facet_filename=input_filename)
    input_filename = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/OutputFull2018/EventSequence/EventSequenceFACET-All.csv"
    parse_event_into_intervals(event_facet_filename=input_filename)
