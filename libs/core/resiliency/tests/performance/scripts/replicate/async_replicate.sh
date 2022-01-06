#!/bin/bash

# Copyright (c) 2019 National Technology & Engineering Solutions of Sandia,
#                    LLC (NTESS).
# Copyright (c) 2019 Nikunj Gupta
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

n_values='3'
exec_times='0 200 500'
cores='32 16 8 4 1'

for n_value in $n_values
do
    for core in $cores
    do
        for exec_time in $exec_times
        do
            for total in {1..10}
            do
                ../../../build/bin/pure_async_for_replicate_benchmarks -t${core} --n-value=${n_value} --exec-time=${exec_time} --error-rate=10000000 >> async_replicate/pure_async_replicate_${core}_${exec_time}_${total}.txt
                ../../../build/bin/async_replicate_benchmarks -t${core} --n-value=${n_value} --exec-time=${exec_time} --error-rate=10000000 >> async_replicate/pure_async_replicate_${core}_${exec_time}_${total}.txt
                ../../../build/bin/async_replicate_validate_benchmarks -t${core} --n-value=${n_value} --exec-time=${exec_time} --error-rate=10000000 >> async_replicate/pure_async_replicate_${core}_${exec_time}_${total}.txt
                ../../../build/bin/async_replicate_vote_benchmarks -t${core} --n-value=${n_value} --exec-time=${exec_time} --error-rate=10000000 >> async_replicate/pure_async_replicate_${core}_${exec_time}_${total}.txt
                ../../../build/bin/async_replicate_vote_validate_benchmarks -t${core} --n-value=${n_value} --exec-time=${exec_time} --error-rate=10000000 >> async_replicate/pure_async_replicate_${core}_${exec_time}_${total}.txt
                echo "done ${exec_time}_${core}_${total}"
            done
        done
    done
done

