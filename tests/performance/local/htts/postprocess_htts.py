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
STASKS = 2
OS_THREADS = 3

LAST_IVAR = OS_THREADS

# Returns the index of the independent variable that we use to differentiate
# datasets (dataset == line on the graph).
def dataset_key(row):
    return (int(row[DELAY]), int(row[TASKS]), int(row[STASKS]))

# Returns a list of all the independent variables. 
def ivars(row):
    return tuple(int(row[x]) for x in range(0, LAST_IVAR + 1))

# Returns a list of all the dependent variables. 
def dvars(row):
    return row[(LAST_IVAR + 1):]

op = OptionParser(usage="%prog [input-data] [output-data] [output-gnuplot-header]")
args = op.parse_args()[1]

if len(args) != 3:
    op.print_help()
    exit(1)

input_data = open(args[0], 'r')
output_data = open(args[1], 'w')
output_header = open(args[2], 'w')

master = {}
legend = []

try:
    while True:
        line = input_data.next()

        # Look for the legend 
        if line[0] == '#':
            if line[1] == '#':
                row = line.split(':')
                legend.append([row[1].strip(), row[2].strip()])
            else:
                print >> output_data, line, 
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
            if sample_size is not len(dvs):
                missing = sample_size - len(dvs)
                print "WARNING: Missing "+str(missing)+" sample(s) for "+\
                      "("+", ".join(str(x) for x in ivs)+")"

        for dv in dvs:
            if number_of_dvars is None:
                number_of_dvars = len(dv)
            else:
                assert number_of_dvars is len(dv)

for i in range(0, LAST_IVAR + 1):
    print >> output_data, '## %i:%s:%s' % (i, legend[i][0], legend[i][1])

    print >> output_header, '%s="%i"' % (legend[i][0], i + 1) 

for i in range(0, (len(legend) - (LAST_IVAR + 1)) * 2, 2):
    i0 = (LAST_IVAR + 1) + i
    i1 = (LAST_IVAR + 1) + (i / 2)

    print >> output_data, '## %i:%s_AVG:%s - Average of %i Samples'\
        % (i0, legend[i1][0], legend[i1][1], sample_size) 
    print >> output_data, '## %i:%s_STD:%s - Standard Deviation'\
        % (i0 + 1, legend[i1][0], legend[i1][1])

    print >> output_header, '%s_AVG="%i"' % (legend[i1][0], i0 + 1) 
    print >> output_header, '%s_STD="%i"' % (legend[i1][0], i0 + 2)

is_first = True

for (key, dataset) in sorted(master.iteritems()):
    if not is_first: 
        print >> output_data
        print >> output_data
    else:
        is_first = False

    print >> output_data, "\"%i μs, %i tasks\"" % (key[DELAY], key[TASKS])

    # iv is a list, dvs is a list of lists.
    for (iv, dvs) in sorted(dataset.iteritems()):
        for e in iv:
            print >> output_data, e,

        for i in range(0, number_of_dvars):
            values = []
            for j in range(0, sample_size):
                values.append(float(dvs[j][i]))

            print >> output_data, mean(values), std(values),

        print >> output_data

