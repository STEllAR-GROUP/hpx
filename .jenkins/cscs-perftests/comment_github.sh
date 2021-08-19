#!/bin/bash -l
set -eux

pushd perftests-reports/reports-comparison

curl \
  -X POST \
  -H "Authorization: token ${GITHUB_TOKEN}" \
  https://api.github.com/repos/STEllAR-GROUP/hpx/pulls/${ghprbPullId}/comments \
  -d "{"body":"dummy comment"}"
  #-d "{"body":"$(cat $index.html)"}"

popd
