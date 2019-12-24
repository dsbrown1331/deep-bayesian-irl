import sys
if len(sys.argv) != 4:
    print("Usage: " + sys.argv[0] + " <infile> <ndecimals> <yscale>")
    print("Try ndecimals = 0 and yscale = 500")
    sys.exit()
import collections
import math

infile = sys.argv[1]
ndecimals = int(sys.argv[2])
yscale = int(sys.argv[3])

pow_val = 1
for x in range(ndecimals):
    pow_val *= 10

with open(infile, "r") as f:
    for line in f:
        data = [int(float(x)*pow_val) for x in line.split(",")]
        counter = collections.Counter(data)
        keys = counter.keys()
        min_key = min(keys)
        max_key = max(keys)
        max_n = -1
        for i in range(min_key, max_key+1):
            if counter[i] > max_n:
                max_n = counter[i]
            #print(counter[i])
        for row in range(math.ceil(max_n/yscale)):
            for i in range(min_key, max_key+1):
                if counter[i]/yscale > int(max_n/yscale)-row:
                    print("#", end="")
                else:
                    print(" ", end="")
            print("")
        strindex = 0
        printedall = False
        while printedall == False:
            printedall = True
            for i in range(min_key, max_key+1):
                st = str(i/pow_val)
                if strindex < len(st):
                    printedall = False
                    print(st[strindex], end="")
                else:
                    print(" ", end="")
            print("")
            strindex += 1
        #print(counter)
