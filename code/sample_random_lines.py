import sys
if len(sys.argv) != 3:
    print("Usage: " + sys.argv[0] + " <infile> <nsamples>")
    sys.exit()
import random
infile = sys.argv[1]
nsamples = int(sys.argv[2])

#-----------------------------------------------------#

nlines = 0
with open(infile, "r") as f:
    for line in f:
        nlines += 1

for _ in range(nsamples):
    sample_spot = random.randint(0, nlines)
    nline = 0
    with open(infile, "r") as f:
        for line in f:
            if nline == sample_spot:
                data = [float(x.strip()) for x in line.split("\t")]
                """
                l1_norm = 0
                for x in data:
                    l1_norm += abs(x)
                print(l1_norm)
                """
                print(" ".join([(f"{1000*x:.0f}".ljust(4)) for x in data]))
                break
            else:
                nline += 1
