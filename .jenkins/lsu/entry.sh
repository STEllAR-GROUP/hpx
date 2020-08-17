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

export configuration_name_with_build_type="${configuration_name}-${build_type,,}"

source .jenkins/lsu/slurm-configuration-${configuration_name}.sh

if [[ -z "${ghprbPullId:-}" ]]; then
    # Set name of branch if not building a pull request
    export git_local_branch=$(echo ${GIT_BRANCH} | cut -f2 -d'/')
    job_name="jenkins-hpx-${git_local_branch}-${configuration_name_with_build_type}"
else
    job_name="jenkins-hpx-${ghprbPullId}-${configuration_name_with_build_type}"

    # Cancel currently running builds on the same branch, but only for pull
    # requests
    scancel --jobname="${job_name}"
fi

# Start the actual build
set +e
sbatch \
    --job-name="${job_name}" \
    --nodes="${configuration_slurm_num_nodes}" \
    --partition="${configuration_slurm_partition}" \
    --time="06:00:00" \
    --output="jenkins-hpx-${configuration_name_with_build_type}.out" \
    --error="jenkins-hpx-${configuration_name_with_build_type}.err" \
    --wait .jenkins/lsu/batch.sh
set -e

# Print slurm logs
echo "= stdout =================================================="
cat jenkins-hpx-${configuration_name_with_build_type}.out

echo "= stderr =================================================="
cat jenkins-hpx-${configuration_name_with_build_type}.err

# Get build status
if [[ "$(cat jenkins-hpx-${configuration_name_with_build_type}-ctest-status.txt)" -eq "0" ]]; then
    github_commit_status="success"
else
    github_commit_status="failure"
fi

if [[ -n "${ghprbPullId:-}" ]]; then
    # Extract just the organization and repo names "org/repo" from the full URL
    github_commit_repo="$(echo $ghprbPullLink | sed -n 's/https:\/\/github.com\/\(.*\)\/pull\/[0-9]*/\1/p')"

    # Get the CDash dashboard build id
    cdash_build_id="$(cat jenkins-hpx-${configuration_name_with_build_type}-cdash-build-id.txt)"

    # Extract actual token from GITHUB_TOKEN (in the form "username:token")
    github_token=$(echo ${GITHUB_TOKEN} | cut -f2 -d':')

    # Set GitHub status with CDash url
    .jenkins/common/set_github_status.sh \
        "${github_token}" \
        "${github_commit_repo}" \
        "${ghprbActualCommit}" \
        "${github_commit_status}" \
        "${configuration_name_with_build_type}" \
        "${cdash_build_id}" \
        "jenkins/lsu"
fi

set -e
exit $(cat jenkins-hpx-${configuration_name_with_build_type}-ctest-status.txt)
