#!/usr/bin/awk -f
#
# This awk script takes the output of an N-body integrator in the form of
# a stream of snapshots that have been output at successive time intervals. 
# It cuts the stream up into individual snapshots, and write each snapshot 
# into an individual file.
# usage: awk -f split_snapshot1.awk snapshots

BEGIN{
    isnap = 0;
    ip = 0;
    print ARGC, ARGV[1];
    filename = ARGV[1];
}
{
    n[isnap] = $1;
    getline;
    t[isnap] = $1;
    fname = "tmp_" filename "." isnap;
    for (i = 0; i < n[isnap]; i++)
    {
        getline;
        print $0 > fname;
    }
    isnap++;
}
