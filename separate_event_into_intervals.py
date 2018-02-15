from facet_event_file import index_dictionary
from datetime import datetime
input_filename = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/EventSequence/EventSequenceFACET.csv"


def parse_event_into_intervals(event_facet_filename):
    intervals = ["Tutorial.csv", "PreScan.csv", "PostScan.csv"]
    start_time = datetime.now()

    with open(event_facet_filename, 'r') as rfile, open(event_facet_filename[:-4]+intervals[0], 'w') as ofile0, open(event_facet_filename[:-4]+intervals[1], 'w') as ofile1, open(event_facet_filename[:-4]+intervals[2], 'w') as ofile2:
        header = rfile.readline()
        ind = index_dictionary(header, [])
        ofile0.write(header)
        ofile1.write(header)
        ofile2.write(header)
        prev_test_subject = ""
        passed_tutorial = False
        passed_scan = False

        for line in rfile:
            line_split = line.replace("\n","").replace("\r","").split(",")
            if len(line_split) > 5:
                current_test_subject = line_split[ind["TestSubject"]]
                if prev_test_subject != current_test_subject:
                    passed_tutorial = False
                    passed_scan = False

                if not passed_tutorial:
                    ofile0.write(line)
                    if line_split[ind["Event"]] == "PlotPoint" and line_split[ind["Name"]] == "IntroFromKim":
                        passed_tutorial = True
                if not passed_scan:
                    ofile1.write(line)
                    if line_split[ind["Event"]] == "PlotPoint" and line_split[ind["Name"]] == "TestObject":
                        passed_scan = True
                if passed_tutorial and passed_scan:
                    ofile2.write(line)

                prev_test_subject = current_test_subject
    minutes_elapsed = (datetime.now() - start_time).total_seconds() / 60.
    print("Successfully split EventSequence into the following files in %.3f minutes:\n\t%s\n\t%s\n\t%s" %
          (minutes_elapsed, event_facet_filename[:-4]+intervals[0], event_facet_filename[:-4]+intervals[1],
           event_facet_filename[:-4] + intervals[2]))

if __name__ == "__main__":
    parse_event_into_intervals(event_facet_filename=input_filename)