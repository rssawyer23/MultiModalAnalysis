# Script for removing excess carriage returns in EventSequence files
import datetime


def parse_script(input_filename, output_filename, show=False):
    """Replace extra carriage return line that occurs on some lines and output new file without such characters"""
    if show:
        print("Started reformatting %s at %s"% (input_filename, datetime.datetime.now()))
    start_time = datetime.datetime.now()
    with open(input_filename, 'rb') as in_file, open(output_filename, 'wb') as o_file:
        fixed_lines = 0
        for line in in_file:
            if b'"\r"' in line:
                fixed_lines += 1
            new_line = line.replace(b'"\r"', b'')
            o_file.write(new_line)
        if show:
            print(fixed_lines)
            print("Fixed %d lines in %s seconds" % (fixed_lines, (datetime.datetime.now() - start_time).total_seconds()))

if __name__ == "__main__":
    #in_filename = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/EventSequence/EventSequence_wGS.csv"
    #out_filename = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/EventSequence/EventSequence_GSP.csv"
    in_filename = "C:/Users/robsc/Documents/NC State/GRAWork/IVHData/OutputJan18Neg/FACET/FACET.csv"
    out_filename = "C:/Users/robsc/Documents/NC State/GRAWork/IVHData/OutputJan18Neg/FACET/FACET.csv"

    assert in_filename != out_filename  # Output filename needs to be a different filename for writing
    parse_script(input_filename=in_filename, output_filename=out_filename, show=True)
