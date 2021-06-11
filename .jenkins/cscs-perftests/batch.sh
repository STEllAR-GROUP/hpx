#!/bin/bash -l

# Copyright (c) 2020 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

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
result=${build_dir}/reports/local-priority-fifo.json
logfile=${build_dir}/reports/jenkins-hpx-${configuration_name}.log

# Load python packages
source /apps/daint/SSL/HPX/virtual_envs/perftests_env/bin/activate

# Things went alright by default
configure_build_errors=0
test_errors=0
plot_errors=0

# Build binaries for performance tests
${perftests_dir}/driver.py -v -l $logfile build -b release \
    -o build --source-dir ${src_dir} --build-dir ${build_dir} -e $envfile \
    -t tests.performance.local.future_overhead_report \
    || { echo 'Build failed'; configure_build_errors=1; exit 1; }


# TODO: make schedulers and other options vary
#for domain in 128 256; do

  # Run performance tests
  ${perftests_dir}/driver.py -v -l $logfile perftest run \
      --local True --scheduling-policy local-priority --run_output $result \
      --extra-opts ' --test-all --repetitions=100' \
      || { echo 'Running failed'; test_errors=1; exit 1; }
  # We add a space before --test-all because of the following issue
  # https://bugs.python.org/issue9334

  # Find references for same configuration (TODO: specify for scheduler etc.)
  reference=${perftests_dir}/perftest/references/daint_default/local-priority-fifo.json

  # Plot comparison of current result with references
  ${perftests_dir}/driver.py -v -l $logfile perftest plot compare \
      -i $reference $result -o ${build_dir}/reports/reports-comparison \
      || { echo 'Plotting failed: performance drop or unknown'; plot_errors=1; exit 1; }
#done

# Dummy ctest to upload the html report of the perftest
set +e
ctest \
    --verbose \
    -S ${src_dir}/.jenkins/cscs-perftests/ctest.cmake \
    -DCTEST_CONFIGURE_EXTRA_OPTIONS="${configure_extra_options}" \
    -DCTEST_BUILD_CONFIGURATION_NAME="${configuration_name}" \
    -DCTEST_SOURCE_DIRECTORY="${src_dir}" \
    -DCTEST_BINARY_DIRECTORY="${build_dir}"
set -e
