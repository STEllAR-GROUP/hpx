#!/bin/bash -l

# Copyright (c) 2020 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Make undefined variables errors, print each command
set -eux

# Clean up directory
rm -f jenkins-hpx*

source .jenkins/cscs/slurm-constraint-${configuration_name}.sh

if [[ -z "${ghprbPullId:-}" ]]; then
    # Set name of branch if not building a pull request
    export git_local_branch=$(echo ${GIT_BRANCH} | cut -f2 -d'/')
    job_name="jenkins-hpx-${git_local_branch}-${configuration_name}"
else
    job_name="jenkins-hpx-${ghprbPullId}-${configuration_name}"

    # Cancel currently running builds on the same branch, but only for pull
    # requests
    scancel --jobname="${job_name}"
fi

# Start the actual build
set +e
sbatch \
    --job-name="${job_name}" \
    --nodes="1" \
    --constraint="${configuration_slurm_constraint}" \
    --partition="cscsci" \
    --time="01:30:00" \
    --output="jenkins-hpx-${configuration_name}.out" \
    --error="jenkins-hpx-${configuration_name}.err" \
    --wait .jenkins/cscs/batch.sh
set -e

# Print slurm logs
echo "= stdout =================================================="
cat jenkins-hpx-${configuration_name}.out

echo "= stderr =================================================="
cat jenkins-hpx-${configuration_name}.err

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
        "jenkins/cscs"
fi

set -e
exit $(cat jenkins-hpx-${configuration_name}-ctest-status.txt)
