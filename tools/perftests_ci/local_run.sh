#!/bin/bash
# Copyright (c)      2021 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# To execute from the build directory

source_dir=${1:-~/projects/hpx_perftests_ci}
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
mkdir -p ${build_dir}/reports
logfile=log_perftest_plot.tmp

hpx_targets=("future_overhead_report_test" "stream_report_test")
hpx_test_options=("--hpx:queuing=local-priority --hpx:threads=4 --test-all \
    --repetitions=100 --futures=500000" \
    "--vector_size=1048576 --hpx:threads=4 --iterations=5000 \
    --warmup_iterations=500")

# Build
${perftests_dir}/driver.py -v -l $logfile build -b release -o build \
    --source-dir ${source_dir} --build-dir ${build_dir} \
    -t "${hpx_targets[@]}"

index=0
result_files=""
# Run and compare for each targets specified
for executable in "${hpx_targets[@]}"
do
    test_opts=${hpx_test_options[$index]}
    result=${build_dir}/reports/${executable}.json
    reference=${perftests_dir}/perftest/references/daint_default/${executable}.json
    logfile_tmp=log_perftest_${executable}.tmp
    result_files+=(${result})
    references_files+=(${reference})

    run_command=("./bin/${executable} ${test_opts}")

    # Run
    ${perftests_dir}/driver.py -v -l $logfile_tmp perftest run --local True \
    --run_output $result --targets-and-opts "${run_command[@]}"

    index=$((index+1))
done

# Plot
${perftests_dir}/driver.py -v -l $logfile perftest plot compare --references \
    ${references_files[@]} --results ${result_files[@]} \
    -o ${build_dir}/reports/reference-comparison
