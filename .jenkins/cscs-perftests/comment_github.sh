#!/bin/bash -l
set -eux

pushd perftests-reports/reports-comparison

# In the order of replacement rule in sed:
# - Remove the image as does not display in github comments (section Details in the report)
# - Escape double quotes for JSON compatibility
# - Escape slashes for JSON compatibility
report=$(cat index.html | \
    sed -e 's:<section class="grid-section"><h2>Details[-a-z0-9<>/"=\ \.]*</section>::Ig' \
        -e 's/"/\\"/g' \
        -e 's/\//\\\//g')

curl \
  -X POST \
  -H "Authorization: token ${GITHUB_TOKEN}" \
  https://api.github.com/repos/STEllAR-GROUP/hpx/issues/${ghprbPullId}/comments \
  -d "{\"body\": \"<details><summary>Performance test report</summary>${report}<\/details>\"}"

popd
