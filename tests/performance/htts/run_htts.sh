#! /bin/bash

# Independent variables are:
#
# * Cores
# * Tasks
# * Delay
#
# Usually we put cores on the X-axis, tasks is fixed, and we split the different
# delays into different data sets. 

usage()
{
    echo "Usage: $0 -b /path/to/benchmark -t max_cores -T 'x y ...' -d 'x y ...' [-s samples] [-f /path/to/options/file]"
    echo
    echo "This script runs a parameter sweep for the HTTS benchmark" 
    echo
    echo "Options:"
    echo "  -b    Path to the benchmark."
    echo "  -t    Maximum number of cores to use." 
    echo "  -T    Space-separated list of task counts to use."
    echo "  -d    Space-separated list of delays to use [microseconds]."
    echo "  -s    Number of samples to collect for each combination of parameters."
    echo "  -f    Path to a file containing options to pass to the benchmark."
}

BENCHMARK=""

MAX_CORES=""

DELAY=""

TASKS=""

SAMPLES=1

OPTIONS_FILE=""

###############################################################################
# Argument parsing
while getopts "b:t:T:d:s:f:h" OPTION; do case $OPTION in
    h)
        usage
        exit 0
        ;;
    b)
        BENCHMARK=$OPTARG
        ;;
    t)
        MAX_CORES=$OPTARG
        ;;
    T)
        TASKS="$OPTARG"
        ;;
    d)
        DELAY="$OPTARG"
        ;;
    s)
        SAMPLES=$OPTARG
        ;;
    f)
        OPTIONS_FILE=$OPTARG
        ;;
    ?)
        usage
        exit 1
        ;;
esac; done

if [ "$BENCHMARK" == "" ] || \
   [ "$MAX_CORES" == "" ] || \
   [ "$DELAY"     == "" ] || \
   [ "$TASKS"     == "" ] 
then
    usage
    exit 1
fi

J=0
for delay in $DELAY; do
    for tasks in $TASKS; do
        for cores in $(bash -c "echo {1..$MAX_CORES}"); do
            for samples in $(bash -c "echo {1..$SAMPLES}"); do
                header="--no-header"
                if [ $J == 0 ]; then
                    header=""
                fi
    
                options=""
                if [ "$OPTIONS_FILE" ]; then
                    options=$(cat $OPTIONS_FILE)
                fi
    
                $BENCHMARK $header --tasks=$tasks --delay=$delay -t$cores $options 
    
                J=$((${J}+1))
            done
        done
    done

    echo; echo
done
 

