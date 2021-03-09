#!/bin/bash -l

# Copyright (c) 2020 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set -ux

# Clean up old artifacts
rm -f ./jenkins-hpx* ./*-Testing

# TMP! now using the envfile var in the python script
source .jenkins/cscs/slurm-constraint-${configuration_name}.sh

#if [[ -z "${ghprbPullId:-}" ]]; then
#    # Set name of branch if not building a pull request
#    export git_local_branch=$(echo ${GIT_BRANCH} | cut -f2 -d'/')
#    job_name="jenkins-hpx-${git_local_branch}-${configuration_name}"
#else
#    job_name="jenkins-hpx-${ghprbPullId}-${configuration_name}"
#
#    # Cancel currently running builds on the same branch, but only for pull
#    # requests
#    scancel --jobname="${job_name}"
#fi

# Args for the pyutils suite
logfile=jenkins-hpx-${configuration_name}.out

orig_src_dir="$(pwd)"
src_dir="/dev/shm/hpx/src"
build_dir="/dev/shm/hpx/build"

# Copy source directory to /dev/shm for faster builds
mkdir -p "${build_dir}"
rsync -r --exclude=.git "${orig_src_dir}" "${src_dir}"

# Tmp! Will be handled by the python script later
envfile=${src_dir}/.jenkins/cscs/env-${configuration_name}.sh

# Copy the perftest utility in the build dir
mkdir -p ${build_dir}/tools/perftests_ci
cp -r ${src_dir}/tools/perftests_ci/* ${build_dir}/tools/perftests_ci

pushd ${build_dir} > /dev/null
# FIXME: we can probably do this step as the other tests and use gridtools
# script only for running and plotting
# build binaries for performance tests
./tools/perftests_ci/driver.py -v -l $logfile build -b release \
    -o build --source-dir ${src_dir} --build-dir ${build_dir} -e $envfile \
    -t tests.performance.local.future_overhead_report \
    || { echo 'Build failed'; exit 1; }

# TODO: make schedulers and other options vary
#for domain in 128 256; do
  # result directory, create if it does not exist yet
  resultdir=${build_dir}/tools/perftests_ci/results
  mkdir -p $resultdir
  result=$resultdir/local-priority-fifo.json

  # run performance tests
  ./tools/perftests_ci/driver.py -v -l $logfile perftest run \
      --scheduling-policy local-priority --run_output $result \
      --extra-opts ' --test-all --repetitions=15' \
      || { echo 'Running failed'; exit 1; }
  # We add a space before --test-all because of the following issue
  # https://bugs.python.org/issue9334

  # create directory for reports
  mkdir reports
  # find references for same configuration
  reference=${src_dir}/tools/perftests_ci/perftest/references/daint_default/local-priority-fifo.json
  # plot comparison of current result with references
  ./tools/perftest_ci/driver.py -v -l $logfile perftest plot compare -i $reference $result -o reports-comparison-$configuration_name || { echo 'Plotting failed'; exit 1; }
#done

# Copy the testing directory for saving as an artifact
cp -r ${build_dir}/Testing ${orig_src_dir}/${configuration_name}-Testing

popd

## Things went wrong by default
#ctest_exit_code=$?
#file_errors=1
#configure_errors=1
#build_errors=1
#test_errors=1
## Temporary as the output files have not been set up
#if [[ -f ${build_dir}/Testing/TAG ]]; then
#    file_errors=0
#    tag="$(head -n 1 ${build_dir}/Testing/TAG)"
#
#    if [[ -f "${build_dir}/Testing/${tag}/Configure.xml" ]]; then
#        configure_errors=$(grep '<Error>' "${build_dir}/Testing/${tag}/Configure.xml" | wc -l)
#    fi
#
#    if [[ -f "${build_dir}/Testing/${tag}/Build.xml" ]]; then
#        build_errors=$(grep '<Error>' "${build_dir}/Testing/${tag}/Build.xml" | wc -l)
#    fi
#
#    if [[ -f "${build_dir}/Testing/${tag}/Test.xml" ]]; then
#        test_errors=$(grep '<Test Status=\"failed\">' "${build_dir}/Testing/${tag}/Test.xml" | wc -l)
#    fi
#fi
#ctest_status=$(( ctest_exit_code + file_errors + configure_errors + build_errors + test_errors ))

# TMP!
ctest_status=0
echo "${ctest_status}" > "jenkins-hpx-${configuration_name}-ctest-status.txt"

# Print slurm logs
echo "= stdout =================================================="
cat jenkins-hpx-${configuration_name}.out

# Get build status
if [[ "$(cat jenkins-hpx-${configuration_name}-ctest-status.txt)" -eq "0" ]]; then
    github_commit_status="success"
else
    github_commit_status="failure"
fi

if [[ -n "${ghprbPullId:-}" ]]; then
    # Extract just the organization and repo names "org/repo" from the full URL
    github_commit_repo="$(echo $ghprbPullLink | sed -n 's/https:\/\/github.com\/\(.*\)\/pull\/[0-9]*/\1/p')"

    # Get the CDash dashboard build id
    cdash_build_id="$(cat jenkins-hpx-${configuration_name}-cdash-build-id.txt)"

    # Extract actual token from GITHUB_TOKEN (in the form "username:token")
    github_token=$(echo ${GITHUB_TOKEN} | cut -f2 -d':')

    # Set GitHub status with CDash url
    .jenkins/common/set_github_status.sh \
        "${github_token}" \
        "${github_commit_repo}" \
        "${ghprbActualCommit}" \
        "${github_commit_status}" \
        "${configuration_name}" \
        "${cdash_build_id}" \
        "jenkins/cscs/perftests"
fi

set -e
exit $(cat jenkins-hpx-${configuration_name}-ctest-status.txt) && $ctest_status
