import sys
if len(sys.argv) != 2:
    print("Usage: " + sys.argv[0] + " <infile>")
    sys.exit()

infile = sys.argv[1]

lines = set()
nlines = 0
with open(infile, "r") as f:
    for line in f:
        if line.strip() != "":
            lines.add(line)
            nlines += 1

print(f"Unique MCMC spots visited: {len(lines):,} out of {nlines:,} recorded locations")
print(f"Jumps occur {len(lines)/nlines:.2%} of the time, or every {nlines/len(lines):.2f} lines")
