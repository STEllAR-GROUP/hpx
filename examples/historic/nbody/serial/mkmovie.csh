#!/bin/csh -f

# A csh script to take the output from a N body integrator, 
# in the form of stream of snapshots that have been output 
# at successive time intervals. As a result of running this 
# script, a gnuplot movie will run at a  rate of one FPS, 
# where successive frames show successive snapshots. 

#usage ./mkmovie snapshots

set snapfile = $1
rm tmp_${snapfile}.*
awk -f snapshot.awk $snapfile
set snapno = 0
set tmpsnap = tmp_${snapfile}
rm tmp1
while ( -e ${tmpsnap}.${snapno} )
echo ${tmpsnap}.${snapno} >> tmp1
@ snapno = $snapno + 1
echo snapno = $snapno
end
awk -f mkmovie1.awk tmp1 > tmp2.gnu
gnuplot tmp2.gnu
