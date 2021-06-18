#!/bin/bash
# Copyright (c)      2021 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# To execute from the build directory

source_dir=~/projects/hpx_perftest_ci
build_dir=$PWD

# Clean old artifacts if any
rm -rf ${build_dir}/reports/reference-comparison

# Setup the environment (libs + python)
source ${source_dir}/.jenkins/cscs-perftests/env-perftests.sh
source /apps/daint/SSL/HPX/virtual_envs/perftests_env/bin/activate

# Variables
mkdir -p ${build_dir}/tools
cp -r ${source_dir}/tools/perftests_ci ${build_dir}/tools
perftests_dir=${build_dir}/tools/perftests_ci
reference=${perftests_dir}/perftest/references/daint_default/local-priority-fifo.json
mkdir -p ${build_dir}/reports
result=${build_dir}/reports/local-priority-fifo.json
logfile=log_perftest.tmp

# Build
${perftests_dir}/driver.py -v -l $logfile build -b release -o build --source-dir ${source_dir} --build-dir ${build_dir} -t tests.performance.local.future_overhead_report

# Run
${perftests_dir}/driver.py -v -l $logfile perftest run --local True \
--scheduling-policy local-priority --run_output $result --extra-opts \
' --test-all --repetitions=100'
# We add a space before --test-all because of the following issue
# https://bugs.python.org/issue9334

# Plot
${perftests_dir}/driver.py -v -l $logfile perftest plot compare -i $reference \
 $result -o ${build_dir}/reports/reference-comparison
