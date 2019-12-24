import sys
if len(sys.argv) != 5:
    print("-- This tool saves a range of columns the last column in a CSV and changes separator to tab --")
    print("Usage: " + sys.argv[0] + " <in_file> <out_file> <start_col> <end_col>")
    sys.exit()

import os
infile = sys.argv[1]
outfile = sys.argv[2]
start_col = int(sys.argv[3])
end_col = int(sys.argv[4])
if os.path.exists(outfile):
    cont = input("Path exists: " + outfile + ". Continue? [y/N] ")
    if len(cont) < 1 or cont[0].upper() != 'Y':
        print("Exiting...")
        sys.exit()

with open(infile, "r") as in_f:
    with open(outfile, "w") as out_f:
        for line in in_f:
            data = line.split(",")
            out_f.write("\t".join(data[start_col:end_col]) + '\n')
