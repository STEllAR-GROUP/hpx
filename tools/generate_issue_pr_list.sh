#!/usr/bin/env bash
#
# Copyright (c) 2019 Mikael Simberg
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file BOOST_LICENSE_1_0.rst or copy at http://www.boost.org/LICENSE_1_0.txt)

# This script generates issue and PR lists for the current release. The output
# is meant to be used in the release notes for each release. It relie on the hub
# command line tool (https://hub.github.com/), jq, and sed.

VERSION_MAJOR=$(sed -n 's/set(HPX_VERSION_MAJOR \(.*\))/\1/p' CMakeLists.txt)
VERSION_MINOR=$(sed -n 's/set(HPX_VERSION_MINOR \(.*\))/\1/p' CMakeLists.txt)
VERSION_SUBMINOR=$(sed -n 's/set(HPX_VERSION_SUBMINOR \(.*\))/\1/p' CMakeLists.txt)
VERSION_FULL_NOTAG=$VERSION_MAJOR.$VERSION_MINOR.$VERSION_SUBMINOR

# hub does not have a sub-command for milestones, but we can list milestones
# using the hub api command instead (based on
# https://github.com/github/hub/issues/2063#issuecomment-472181266)
github_milestones() {
  hub api --cache 3600 graphql -f query='
    {
      repository(owner: "{owner}", name: "{repo}") {
        milestones(first: 100, states: OPEN, orderBy: {field:CREATED_AT, direction:DESC}) {
          edges {
            node {
              title
              number
            }
          }
        }
      }
    }
  ' | jq -r '.data.repository.milestones.edges[].node | [.number,.title] | @tsv'
}

milestone_id_from_version() {
    github_milestones | grep "${1}" | cut -f1
}

VERSION_MILESTONE_ID=$(milestone_id_from_version "${VERSION_FULL_NOTAG}")

# echo "Closed issues"
# echo "============="

# hub issue --state=closed --milestone="${VERSION_MILESTONE_ID}" --format="* :hpx-issue:\`%I\` - %t%n"

echo ""
echo "Closed pull requests"
echo "===================="

# The hub pr list command does not allow filtering by milestone like hub issue.
# However, it lets us print the milestone for each PR. So we print every PR with
# a milestone, filter out the unwanted PRs, and remove the printed milestone
# from every PR instead.
hub pr list --state=closed --format="[%Mn]* :hpx-pr:\`%I\` - %t%n" |
    sed -n "s/^\[${VERSION_MILESTONE_ID}\]\(.*\)/\1/p"
