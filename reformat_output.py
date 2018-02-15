# Script for removing excess carriage returns in EventSequence files


def parse_script(input_filename, output_filename):
    with open(input_filename, 'rb') as in_file, open(output_filename, 'wb') as o_file:
        for line in in_file:
            new_line = line.replace(b'"\r"', b'')
            o_file.write(new_line)

if __name__ == "__main__":
    in_filename = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/EventSequence/EventSequence_wGS.csv"
    out_filename = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/EventSequence/EventSequence_GSP.csv"
    parse_script(input_filename=in_filename, output_filename=out_filename)
