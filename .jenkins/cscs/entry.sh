#!/bin/bash -l

# Copyright (c) 2020 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Make undefined variables errors, print each command
set -eux

# Clean up old artifacts
rm -f ./jenkins-hpx* ./*-Testing

export configuration_name_with_build_type="${configuration_name}-${build_type,,}"

source .jenkins/cscs/slurm-constraint-${configuration_name}.sh

if [[ -z "${ghprbPullId:-}" ]]; then
    # Set name of branch if not building a pull request
    export git_local_branch=$(echo ${GIT_BRANCH} | cut -f2 -d'/')
    job_name="jenkins-hpx-${git_local_branch}-${configuration_name_with_build_type}"
else
    job_name="jenkins-hpx-${ghprbPullId}-${configuration_name_with_build_type}"

    # Cancel currently running builds on the same branch, but only for pull
    # requests
    scancel --account="djenkssl" --jobname="${job_name}"
fi

# Start the actual build
set +e
sbatch \
    --job-name="${job_name}" \
    --nodes="1" \
    --constraint="${configuration_slurm_constraint}" \
    --partition="cscsci" \
    --account="djenkssl" \
    --time="02:00:00" \
    --output="jenkins-hpx-${configuration_name_with_build_type}.out" \
    --error="jenkins-hpx-${configuration_name_with_build_type}.err" \
    --wait .jenkins/cscs/batch.sh

# Print slurm logs
echo "= stdout =================================================="
cat jenkins-hpx-${configuration_name_with_build_type}.out

echo "= stderr =================================================="
cat jenkins-hpx-${configuration_name_with_build_type}.err

# Get build status
status_file="jenkins-hpx-${configuration_name_with_build_type}-ctest-status.txt"
if [[ -f "${status_file}" && "$(cat ${status_file})" -eq "0" ]]; then
    github_commit_status="success"
else
    github_commit_status="failure"
fi

if [[ -n "${ghprbPullId:-}" ]]; then
    # Extract just the organization and repo names "org/repo" from the full URL
    github_commit_repo="$(echo $ghprbPullLink | sed -n 's/https:\/\/github.com\/\(.*\)\/pull\/[0-9]*/\1/p')"

    # Get the CDash dashboard build id
    cdash_build_id="$(cat jenkins-hpx-${configuration_name_with_build_type}-cdash-build-id.txt)"

    # Set GitHub status with CDash url
    .jenkins/common/set_github_status.sh \
        "${GITHUB_TOKEN}" \
        "${github_commit_repo}" \
        "${ghprbActualCommit}" \
        "${github_commit_status}" \
        "${configuration_name_with_build_type}" \
        "${cdash_build_id}" \
        "jenkins/cscs"
fi

set -e
exit $(cat ${status_file})
