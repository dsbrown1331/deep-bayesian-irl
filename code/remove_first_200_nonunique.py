import sys
if len(sys.argv) != 3:
    print("Usage: " + sys.argv[0] + " <infile> <outfile>")
    sys.exit()

infile = sys.argv[1]
outfile = sys.argv[2]
import os
if os.path.exists(outfile):
    yn = input("Outfile exists. Continue? [y/N] ")
    if len(yn) < 1 or yn[0].upper() != "Y":
        sys.exit()

with open(infile, "r") as f_in:
    with open(outfile, "w") as f_out:
        linesfound = set()
        for line in f_in:
            if len(linesfound) >= 200:
                f_out.write(line)
            else:
                linesfound.add(line)
                
