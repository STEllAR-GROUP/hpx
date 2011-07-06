#!/usr/bin/awk -f
#
# This script takes a list of filenames, each containing one Nbody 
# snapshot. The files are produced by the awk script split_snapshot.awk
# from an NBody snapshot stream. The commoands are then generated for 
# gnuplot to show a movie, at a rate of 1 frame per second. 
#
# Usage: intended to be used by a higher level script. 
BEGIN{
    print "set size ratio 1";
#    print "set xrange [-3:3]";
#    print "set yrange [-3:3]";
}
{
    print "splot \"" $1 "\" using 1:2:3 notitle";
#    print "splot \"" $1 "\" using 1:2:3 notitle pointtype 1 pointsize 2";
    print "pause 1";
}
END{
    print "pause -1 \"Hit return to exit\"";
}
