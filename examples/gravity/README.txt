///////////////////////////////////////////////////////////////////////////////
/////////////////////////////       Gravity       /////////////////////////////
///////////////////////////////////////////////////////////////////////////////

Gravity is a set of two solutions to the n-body problem. The first solution 
found in the gravity_future directory only takes advantage of HPX futures 
for its parallization. This program will not run in distributed. The other
solution is a more recent adaptation of gravity_future. Gravity_dataflow 
is the newest addition to the Gravity family. Utilizing HPX dataflows and 
futures, gravity_dataflow promises to be a much more efficient solution the 
n-body problem.  Gravity_dataflow also has the most promise of being able to 
run in distrubuted.

Both versions of gravity use the configuration file "gravconfig.conf" (found in 
this directory to initialize their variables. Also both take HDF5 files as 
their input. (It should be noted that if a user places "--debug" as a command 
line option both programs will print a .txt verson of their output as well.)
To build the and run the programs issue the following commands:

gravity_future:

     make examples.gravity_hpx
     ./bin/gravity_hpx --config-file (path to config file) 
                       --hpx:threads (# of threads)

gravity_dataflow:

     make examples.gravity_dataflow
     ./bin/gravity_dataflow --config-file (path to config file) 
                            --hpx:threads (# of threads)

///////////////////////////////////////////////////////////////////////////////
