#!/bin/bash -l

# Copyright (c) 2020 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set -eux

# Computes the status of the job and store the artifacts
status_computation_and_artifacts_storage() {
    ctest_exit_code=$?
    ctest_status=$(( ctest_exit_code + configure_build_errors + test_errors + plot_errors ))

    # Copy the testing directory for saving as an artifact
    cp -r ${build_dir}/Testing ${orig_src_dir}/${configuration_name}-Testing
    cp -r ${build_dir}/reports ${orig_src_dir}/${configuration_name}-reports

    echo "${ctest_status}" > "jenkins-hpx-${configuration_name}-ctest-status.txt"
    exit $ctest_status
}

trap "status_computation_and_artifacts_storage" EXIT

orig_src_dir="$(pwd)"
src_dir="/dev/shm/hpx/src"
build_dir="/dev/shm/hpx/build"

# Copy source directory to /dev/shm for faster builds
mkdir -p "${build_dir}"
cp -r "${orig_src_dir}" "${src_dir}"
# Args for the pyutils suite
envfile=${src_dir}/.jenkins/cscs-perftests/env-${configuration_name}.sh
# Copy the perftest utility in the build dir
mkdir -p ${build_dir}/tools
cp -r ${src_dir}/tools/perftests_ci ${build_dir}/tools

# Variables
perftests_dir=${build_dir}/tools/perftests_ci
mkdir -p ${build_dir}/reports
logfile=${build_dir}/reports/jenkins-hpx-${configuration_name}.log

# Load python packages
source /apps/daint/SSL/HPX/virtual_envs/perftests_env/bin/activate

# Things went alright by default
configure_build_errors=0
test_errors=0
plot_errors=0

hpx_targets=("future_overhead_report_test" "stream_report_test")
hpx_test_options=("--hpx:queuing=local-priority --hpx:threads=4 --test-all \
    --repetitions=100 --futures=500000" \
    "--vector_size=1048576 --hpx:threads=4 --iterations=5000 \
    --warmup_iterations=500")

# Build binaries for performance tests
${perftests_dir}/driver.py -v -l $logfile build -b release -o build \
    --source-dir ${src_dir} --build-dir ${build_dir} -e $envfile \
    -t "${hpx_targets[@]}" \
    || { echo 'Build failed'; configure_build_errors=1; exit 1; }

index=0
result_files=""

# Run and compare for each targets specified
for executable in "${hpx_targets[@]}"
do
  test_opts=${hpx_test_options[$index]}
  result=${build_dir}/reports/${executable}.json
  reference=${perftests_dir}/perftest/references/daint_default/${executable}.json
  result_files+=(${result})
  references_files+=(${reference})
  logfile_tmp=log_perftests_${executable}.tmp

  run_command=("./bin/${executable} ${test_opts}")

  # TODO: make schedulers and other options vary

  # Run performance tests
  ${perftests_dir}/driver.py -v -l $logfile_tmp perftest run --local True \
      --run_output $result --targets-and-opts "${run_command[@]}" \
      || { echo 'Running failed'; test_errors=1; exit 1; }

  index=$((index+1))
done

# Plot comparison of current result with references
${perftests_dir}/driver.py -v -l $logfile perftest plot compare --references \
    ${references_files[@]} --results ${result_files[@]} \
    -o ${build_dir}/reports/reports-comparison \
    || { echo 'Plotting failed: performance drop or unknown'; plot_errors=1; exit 1; }

# Dummy ctest to upload the html report of the perftest
set +e
ctest \
    --verbose \
    -S ${src_dir}/.jenkins/cscs-perftests/ctest.cmake \
    -DCTEST_BUILD_CONFIGURATION_NAME="${configuration_name}" \
    -DCTEST_SOURCE_DIRECTORY="${src_dir}" \
    -DCTEST_BINARY_DIRECTORY="${build_dir}"
set -e
