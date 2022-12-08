#!/usr/bin/python3
#Scripted by Syun
#modified version - Shyeon
#2022.03.04

import numpy as np
import sys
import os
import random
import re

if len(sys.argv) < 3:
    print("---------------------------------------------------------")
    print("\[ERROR\] The number of arguments is less than 3")
    print("-How to use-")
    print("./rand_height.tcl {LEF File} {Output File} {random ratio}")
    print("---------------------------------------------------------")
    sys.exit()

print("------------------------------------------")
print("Generate Mixed-Height Standard Cells")
print("- Select 15% of cells randomly")
print("- Multiply cell height (x2 x3 x4)")
print("------------------------------------------")
print("InputFile:  {}".format(sys.argv[1]))
print("OutputFile: {}".format(sys.argv[2]))
print("------------------------------------------")
print("")

Filein  = sys.argv[1]
Fileout = sys.argv[2]
ratio   = sys.argv[3]

if float(ratio) > 1.0:
    print("Ratio should be between 0 and 1")
    sys.exit()

################################################3

inFile  = open(Filein, "r")
outFile = open(Fileout, "w")

MacroNum  = 0
twice     = 0
triple    = 0
quad      = 0
while True:
    line = inFile.readline()
    if not line or line=="":
        break
    outFile.write(line)
    if re.match(r"MACRO", line):
        MacroNum += 1
        while True:
            line = inFile.readline()
            if re.match(r"[^\S]+SIZE", line):
                n = random.random()
                if n < float(ratio):
                    elem   = line.split()
                    width  = float(elem[1])
                    height = float(elem[3])
                    n2 = random.random()
                    if n2 < 0.6:
                        twice += 1
                        width  = width/2.0
                        height = height*2.0
                    elif 0.6 < n2 < 0.9:
                        triple += 1
                        width  = width/3.0
                        height = height*3.0
                    else:
                        quad += 1
                        width = width/4.0
                        height = height*4.0
                    outFile.write("  SIZE %.2f BY %.2f ;\n" % (width, height))
                else:
                    outFile.write(line)
                break
            else:
                outFile.write(line)
        
inFile.close()
outFile.close()


print("Total number of macro cells:    {}".format(MacroNum))
print("Modified number of macro cells: {}".format(twice+triple))
print("  - Num. of X2 height cells:    {}".format(twice))
print("  - Num. of X3 height cells:    {}".format(triple))
print("  - Num. of X4 hegiht cells:    {}".format(quad))
print("Ratio of multi-height cells:    {0:.1f}%".format(float((twice+triple)/MacroNum)*100.0))
print("-- Program done! -------------------")
