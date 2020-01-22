import sys
if len(sys.argv) != 5:
    print("Usage: " + sys.argv[0] + " <infile> <outfolder> <ndecimals> <yscale>")
    print("Try ndecimals = 0 and yscale = 500")
    sys.exit()
import os
infile = sys.argv[1]
outfolder = sys.argv[2]
ndecimals = int(sys.argv[3])
yscale = float(sys.argv[4])
os.mkdir(outfolder)

import collections
import math
import png
import numpy as np

pow_val = 1
for x in range(ndecimals):
    pow_val *= 10

with open(infile, "r") as f:
    nline = 0
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
        data_arr = np.zeros(((math.ceil(max_n/yscale)+1), (max_key-min_key+1)*3), dtype=np.uint8)
        for row in range(math.ceil(max_n/yscale)+1):
            for i in range(min_key, max_key+1):
                if counter[i]/yscale > int(max_n/yscale)-row:
                    data_arr[row][(i-min_key)*3+0] = 10
                    data_arr[row][(i-min_key)*3+1] = 200
                    data_arr[row][(i-min_key)*3+2] = 255
                else:
                    if i > 0:
                        data_arr[row][(i-min_key)*3+0] = 255
                        data_arr[row][(i-min_key)*3+1] = 255
                        data_arr[row][(i-min_key)*3+2] = 255
                    elif i == 0:
                        data_arr[row][(i-min_key)*3+0] = 230
                        data_arr[row][(i-min_key)*3+1] = 230
                        data_arr[row][(i-min_key)*3+2] = 230
                    else:
                        data_arr[row][(i-min_key)*3+0] = 250
                        data_arr[row][(i-min_key)*3+1] = 250
                        data_arr[row][(i-min_key)*3+2] = 250
                    #print(" ", end="")
            #print("")
        """
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
        """
        print("Saving image " + str(nline) + ".png")
        png.from_array(data_arr, mode="rgb").save(outfolder + "/" + str(nline) + '.png')
        #print(counter)
        nline += 1
