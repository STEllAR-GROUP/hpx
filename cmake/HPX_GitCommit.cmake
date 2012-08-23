# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_GITCOMMIT_LOADED TRUE)

include(HPX_Include)

hpx_include(Message)

execute_process(
  COMMAND "git" "log" "--pretty=%H" "-1" "${hpx_SOURCE_DIR}"
  OUTPUT_VARIABLE GIT_COMMIT ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

if("${GIT_COMMIT}" STREQUAL "None")
  hpx_warn("git.commit" "GIT commit not found (set to 'unknown').")
  set(GIT_COMMIT "unknown")
else()
  hpx_info("git.commit" "GIT commit is ${GIT_COMMIT}.")
  add_definitions("-DHPX_GIT_COMMIT=\"${GIT_COMMIT}\"")
endif()

