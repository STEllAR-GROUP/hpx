#  Copyright (c) 2014 John Biddiscombe
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#!/usr/bin/python
import optparse
import itertools
from io import StringIO
import csv
import os
import re
import glob
import math
import numpy
from numpy import array
import matplotlib
from plot_utils import *

workdir = os.getcwd()

#----------------------------------------------------------------------------
if len(args) == 0 :
    print("No input CSV file given")
    exit(0)

#----------------------------------------------------------------------------
# to plot something not already included, add it to this list
T_RW_B_NB = {}
T_B_RW_NB = {}

#----------------------------------------------------------------------------
#
# read results data in and generate arrays/maps of values
# for each parcelport, threadcount, blocksize, ...
for csvfile in args :

    print("\nReading file ", csvfile)

    # output file path for svg/png
    base = os.path.splitext(csvfile)[0]
    # empty list of graphs we will be fill for exporting
    graphs_to_save = []

    # open the CSV file
    with open(csvfile) as f:
      io = StringIO(f.read().replace(':', ','))
      reader = csv.reader(io)

      # loop over the CSV file lines,
      # if the CSV output is changed for the test, these offsets will need to be corrected
      for row in reader:
          readflag = row[1].strip() in ("read")
          Network  = row[3]
          Nodes    = int(row[5])
          Threads  = int(row[7])
          IOPsize  = int(row[11])
          IOPs     = float(row[13])
          BW       = float(row[15])/1024.0
          if (BW==0.0) :
            BW = 1.0
          print("read=%i Network=%s Nodes=%4i Threads=%3i IOPsize=%9i IOPs=%6.1f BW=%6.1f"  % (readflag, Network, Nodes, Threads, IOPsize, IOPs, BW))

          # we use a map structure 3 deep with an array at the leaf,
          # this allows us to store param1, param2, param3, {x,y}
          # combinations of params cen be plotted against each other
          # by rearranging the map levels and {x,y} vars.
          insert_safe(T_RW_B_NB, Threads, readflag, IOPsize, [Nodes,BW])
          insert_safe(T_B_RW_NB, Threads, IOPsize, readflag, [Nodes,BW])

average_map(T_RW_B_NB)
average_map(T_B_RW_NB)

##-------------------------------------------------------------------
## PLOT x-axis{Nodes}, y-axis{BW},
fig_T_RW_B_NB  = plot_configuration(T_RW_B_NB,
  ["Threads", "Mode", "Block size"],
  ["Nodes", "BW GB/s"],
  lambda x: "Read" if (x==1) else "Write",      # Plot title
  lambda x: sizeof_bytes(x),                    # Legend text
  lambda x: "Threads = " + str(int(x)),                        # legend title
  lambda x,pos: str(int(x)),                    # X Axis labels
  [[2,0,12,0.0], [2,0,13,0.0]],                 # minmax (base, min, max, padding)
  [0.0, 0.0]                                    # legend offset
  )
graphs_to_save.append([fig_T_RW_B_NB,"fig_T_RW_B_NB"])

##-------------------------------------------------------------------
## PLOT x-axis{Nodes}, y-axis{BW},
fig_T_B_RW_NB  = plot_configuration(T_B_RW_NB,
  ["Threads", "Block size", "Mode"],
  ["Nodes", "BW GB/s"],
  lambda x: sizeof_bytes(x),                    # Plot title
  lambda x: "Read" if (x==1) else "Write",                  # Legend text
  lambda x: "Threads = " + str(int(x)),                        # legend title
  lambda x,pos: str(int(x)),                    # X Axis labels
  [[2,0,12,0.0], [2,0,13,0.0]],                 # minmax (base, min, max, padding)
  [0.0, 0.0]                                    # legend offset
  )
graphs_to_save.append([fig_T_B_RW_NB,"fig_T_B_RW_NB"])

##-------------------------------------------------------------------
# save plots to png and svg
for fig in graphs_to_save:
  svg_name = base + "." + fig[1] + ".svg"
  png_name = base + "." + fig[1] + ".png"
  print("Writing %s" % svg_name)
  fig[0].savefig(svg_name)
  #fig[0].savefig(png_name)

#-------------------------------------------------------------------
