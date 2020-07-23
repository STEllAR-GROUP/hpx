#!/bin/bash -l

# Copyright (c) 2020 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set -eux

github_token=${1}
commit_repo=${2}
commit_sha=${3}
commit_status=${4}
configuration_name=${5}
build_id=${6}
context=${7}

curl --verbose \
    --request POST \
    --url "https://api.github.com/repos/${commit_repo}/statuses/${commit_sha}" \
    --header 'Content-Type: application/json' \
    --header "authorization: Bearer ${github_token}" \
    --data "{ \"state\": \"${commit_status}\", \"target_url\": \"https://cdash.cscs.ch/buildSummary.php?buildid=${build_id}\", \"description\": \"Jenkins\", \"context\": \"${context}/${configuration_name}\" }"
