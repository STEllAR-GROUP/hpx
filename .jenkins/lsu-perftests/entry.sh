#!/bin/bash -l

# Copyright (c) 2020 ETH Zurich
# Copyright (c) 2022 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Make undefined variables errors, print each command
set -eux

status_file="jenkins-hpx-${configuration_name}-ctest-status.txt"
report_file="${configuration_name}-reports/reference-comparison/index.html"
changed_file="${configuration_name}-reports/reference-comparison/changed.txt"
rm -f "jenkins-hpx-${configuration_name}.out" \
    "jenkins-hpx-${configuration_name}.err" "${status_file}" "${report_file}" \
    "${changed_file}"

source .jenkins/common/slurm.sh

source .jenkins/lsu-perftests/slurm-constraint-${configuration_name}.sh

if [[ -z "${ghprbPullId:-}" ]]; then
    # Set name of branch if not building a pull request
    export git_local_branch=$(echo ${GIT_BRANCH} | cut -f2 -d'/')
    job_name="jenkins-hpx-${git_local_branch}-${configuration_name}"
else
    job_name="jenkins-hpx-${ghprbPullId}-${configuration_name}"

    # Cancel currently running builds on the same branch, but only for pull
    # requests. Old jobs that are slow to go away must not stop this build,
    # or the lane ends here without running or reporting anything.
    if ! hpx_slurm_cancel_previous "${job_name}"; then
        echo "Warning: could not clear previous Slurm jobs for ${job_name}," \
            "starting this build anyway" >&2
    fi
fi

# delay things for a random amount of time
sleep $[(RANDOM % 10) + 1].$[(RANDOM % 10)]s

# Run on the first of the nodes that is idle, as the job needs a node for
# itself; wait for the first one if none is. This comes after cancelling the
# previous jobs of the pull request, so that the node they ran on counts as
# idle. The comparison uses the references of the node the job runs on, so the
# nodes that have references come first.
references_dir="tools/perftests_ci/perftest/references"
candidate_nodes=""
for node in ${configuration_slurm_nodes}; do
    if [[ -e "${references_dir}/${node}.tar.gz" ]]; then
        candidate_nodes="${candidate_nodes} ${node}"
    fi
done
for node in ${configuration_slurm_nodes}; do
    if [[ ! -e "${references_dir}/${node}.tar.gz" ]]; then
        candidate_nodes="${candidate_nodes} ${node}"
    fi
done
idle_nodes=$(timeout --kill-after=5s 30s sinfo --noheader --node-oriented \
    --partition="${configuration_slurm_partition}" --states=idle \
    --format=%N) || idle_nodes=""
configuration_slurm_nodelist=""
for node in ${candidate_nodes}; do
    if grep -qx "${node}" <<< "${idle_nodes}"; then
        configuration_slurm_nodelist="${node}"
        break
    fi
done
if [[ -z "${configuration_slurm_nodelist}" ]]; then
    configuration_slurm_nodelist="${configuration_slurm_nodes%% *}"
fi
echo "Running the performance tests on ${configuration_slurm_nodelist}"
# The report names the node (see report_common.sh)
export PERFTEST_NODE="${configuration_slurm_nodelist}"

# A build triggered by the comment "retest lsu perf full" reports every
# measurement, and comments on the pull request, regardless of how much they
# changed (see perftests_plot.py). The trigger phrase of the Jenkins job has to
# accept the comment.
trigger_comment="${ghprbCommentBody:-}"
if [[ "${trigger_comment,,}" =~ retest([[:space:]]+lsu)?([[:space:]]+perf)?[[:space:]]+full ]]; then
    echo "Full performance report requested by a pull request comment"
    export HPX_PERFTESTS_REPORT_ALL=1
fi

# The job needs its node exclusively, so it waits in the queue for as long as
# another job runs there. Fail after a few hours instead of keeping the pull
# request pending for up to a day (the default of hpx_slurm_run).
export HPX_SLURM_QUEUE_TIMEOUT="${HPX_SLURM_QUEUE_TIMEOUT:-3h}"

# Start the actual build
set +e
hpx_slurm_run "${HPX_SLURM_TIMEOUT:-4h}" \
    --verbose --verbose --verbose --verbose \
    --exclusive \
    --job-name="${job_name}" \
    --nodes="1" \
    --partition="${configuration_slurm_partition}" \
    --nodelist="${configuration_slurm_nodelist}" \
    --time="03:00:00" \
    --output="jenkins-hpx-${configuration_name}.out" \
    --error="jenkins-hpx-${configuration_name}.err" \
    .jenkins/lsu-perftests/batch.sh
slurm_status=$?

# Print slurm logs
echo "= stdout =================================================="
cat jenkins-hpx-${configuration_name}.out

echo "= stderr =================================================="
cat jenkins-hpx-${configuration_name}.err

# Comment on a pull request only if a measurement got more than 5% faster or
# slower than its reference (see perftests_plot.py). Failures don't produce a
# comment, they turn the status red.
if [[ -n "${ghprbPullId:-}" && -s "${report_file}" && -s "${changed_file}" ]]; then
    # Post the report without plots if they can't be uploaded
    if ! PERFTEST_IMAGES_URL="$(./.jenkins/lsu-perftests/upload_images.sh \
        "$(dirname "${report_file}")" "${ghprbPullId}" "${BUILD_NUMBER}")"; then
        echo "Warning: could not upload the plots of the report" >&2
        PERFTEST_IMAGES_URL=""
    fi
    PERFTEST_IMAGES_URL="${PERFTEST_IMAGES_URL}" \
        ./.jenkins/lsu-perftests/comment_github.sh
fi

# Set GitHub status with CDash url
if [[ "${slurm_status}" -eq 0 && -s "${status_file}" &&
    "$(cat "${status_file}")" == 0 ]]; then
    github_commit_status="success"
else
    github_commit_status="failure"
fi
cdash_build_id_file="jenkins-hpx-${configuration_name}-cdash-build-id.txt"
cdash_build_id=""
if [[ -s "${cdash_build_id_file}" ]]; then
    cdash_build_id="$(cat "${cdash_build_id_file}")"
fi

# Do not print the token passed to the status helper.
set +x
if [[ -z "${ghprbPullId:-}" ]]; then
    .jenkins/common/set_github_status.sh \
        "${GITHUB_TOKEN}" \
        "TheHPXProject/hpx" \
        "${GIT_COMMIT}" \
        "${github_commit_status}" \
        "${configuration_name}" \
        "${cdash_build_id}" \
        "jenkins/lsu-perftests"
else
    # Extract just the organization and repo names "org/repo" from the full URL
    github_commit_repo="$(echo $ghprbPullLink | sed -n 's/https:\/\/github.com\/\(.*\)\/pull\/[0-9]*/\1/p')"

    .jenkins/common/set_github_status.sh \
        "${GITHUB_TOKEN}" \
        "${github_commit_repo}" \
        "${ghprbActualCommit}" \
        "${github_commit_status}" \
        "${configuration_name}" \
        "${cdash_build_id}" \
        "jenkins/lsu-perftests"
fi
set -x

set -e
if [[ "${slurm_status}" -ne 0 ]]; then
    exit "${slurm_status}"
fi
if [[ ! -s "${status_file}" ]]; then
    echo "Missing performance test status: ${status_file}" >&2
    exit 1
fi
exit "$(cat "${status_file}")"
