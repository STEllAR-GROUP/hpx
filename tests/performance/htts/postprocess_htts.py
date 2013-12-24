#! /usr/bin/env python
# -*- coding: utf-8 -*-

from math import sqrt
from sys import exit

from optparse import OptionParser

from numpy import std, mean

# 0: Delay [micro-seconds] - Independent Variable
# 1: Tasks - Independent Variable
# 2: OS-threads - Independent Variable
# 3: Total Walltime [seconds]
# 4+: Counters

DELAY = 0
TASKS = 1
OS_THREADS = 2

LAST_IVAR = OS_THREADS

# Returns the index of the independent variable that we use to differentiate
# datasets (dataset == line on the graph).
def dataset_key(row):
    return int(row[DELAY])

# Returns a list of all the independent variables. 
def ivars(row):
    return tuple(int(row[x]) for x in range(0, LAST_IVAR + 1))

# Returns a list of all the dependent variables. 
def dvars(row):
    return row[(LAST_IVAR + 1):]

op = OptionParser(usage="%prog [file]")
args = op.parse_args()[1]

if len(args) != 1:
    op.print_help()
    exit(1)

f = open(args[0], 'r')

master = {}
legend = []

try:
    while True:
        line = f.next()

        # Look for the legend 
        if line[0] == '#':
            if line[1] == '#':
                row = line.split(':')
                legend.append(row[1].strip())
            else:
                print line,
            continue   

        # Look for blank lines
        if line == "\n":
            continue

        row = line.split()

        if not dataset_key(row) in master:
            master[dataset_key(row)] = {}

        if not ivars(row) in master[dataset_key(row)]:
            master[dataset_key(row)][ivars(row)] = []

        master[dataset_key(row)][ivars(row)].append(dvars(row))  

except StopIteration:
    pass

sample_size = None
number_of_dvars = None

for (key, dataset) in sorted(master.iteritems()):
    for (ivs, dvs) in sorted(dataset.iteritems()):
        if sample_size is None:
            sample_size = len(dvs)
        else:
            assert sample_size is len(dvs) 

        for dv in dvs:
            if number_of_dvars is None:
                number_of_dvars = len(dv)
            else:
                assert number_of_dvars is len(dv)

for i in range(0, LAST_IVAR + 1):
    print '## %i: %s' % (i, legend[i]) 
for i in range(0, (len(legend) - (LAST_IVAR + 1)) * 2, 2):
    i0 = (LAST_IVAR + 1) + i
    i1 = (LAST_IVAR + 1) + (i / 2)
    print '## %i: %s - Average of %i Samples' % (i0, legend[i1], sample_size)
    print '## %i: %s - Standard Deviation' % (i0 + 1, legend[i1])

is_first = True

for (key, dataset) in sorted(master.iteritems()):
    if not is_first: 
        print
        print
    else:
        is_first = False

    print "\"%i μs\"" % key
        
    # iv is a list, dvs is a list of lists.
    for (iv, dvs) in sorted(dataset.iteritems()):
        for e in iv:
            print e,

        for i in range(0, number_of_dvars):
            values = []
            for j in range(0, sample_size):
                values.append(float(dvs[j][i]))

            print mean(values), std(values),

        print

