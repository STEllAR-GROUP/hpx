#!/bin/bash

SAMPLES=100
PAYLOAD=0
SEED=4171

K=0
for i in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000;
do
    for samples in $(bash -c "echo {1..$SAMPLES}")
    do
        header="--no-header"
        if [ $K == 0 ]; then
            header=""
        fi

        bin/coroutines_call_overhead $header --payload=$PAYLOAD --iterations=10000000 --contexts=$i --seed=$SEED \
            --counter TLB_DM,'/arithmetics/add@/papi{locality#0/worker-thread#*}/PAPI_TLB_DM' \
            --counter LD_INS,'/arithmetics/add@/papi{locality#0/worker-thread#*}/PAPI_LD_INS' \
            --counter SR_INS,'/arithmetics/add@/papi{locality#0/worker-thread#*}/PAPI_SR_INS' \
            --counter TOT_INS,'/counters/arithmetics/add@/papi{locality#0/worker-thread#*}/PAPI_TOT_INS'

        K=$((${J}+1))
    done
done

