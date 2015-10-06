There are two 1d_stencil examples that use runtime adaptation.

The first one is 1d_stencil_4_repart - it will repartition the blocks to adapt
the block size given the problem size and thread count. To run that example,
configure HPX with APEX:

cmake ... \
-DHPX_WITH_APEX=TRUE \
-DAPEX_WITH_ACTIVEHARMONY=TRUE \
-DACTIVEHARMONY_ROOT=<path-to-activeharmony> \
...

Then to execute the example:

export HARMONY_HOME=$HOME/install/activeharmony/4.5
./build.tcmalloc/bin/1d_stencil_4_repart --nx=100000 --nt=500 --nr=10


The other example uses throttling to achieve the highest throughput by 
minimizing the thread queue length.  To execute that example:

export APEX_THROTTLE_CONCURRENCY=1
export APEX_THROTTLE_ENERGY=1
export APEX_POLICY=1
export APEX_THROTTLING_MAX_THREADS=20

./build.tcmalloc/bin/1d_stencil_4_throttle --nx 100000 --np 1000 --nt 450 \
--hpx:queuing=throttle --hpx:threads 20
