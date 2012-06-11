# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_SVNREVISION_LOADED TRUE)

include(HPX_Include)

hpx_include(Message)

if(MSVC)
  execute_process(
    COMMAND "cmd" "/c" "${hpx_SOURCE_DIR}/python/scripts/hpx_svnversion.py" "${hpx_SOURCE_DIR}"
    OUTPUT_VARIABLE SVN_REVISION ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
else()
  execute_process(
    COMMAND "${hpx_SOURCE_DIR}/python/scripts/hpx_svnversion.py" "${hpx_SOURCE_DIR}"
    OUTPUT_VARIABLE SVN_REVISION ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

if("${SVN_REVISION}" STREQUAL "None")
  hpx_warn("svn.revision" "SVN revision not found (set to 'unknown').")
  set(SVN_REVISION "unknown")
else()
  hpx_info("svn.revision" "SVN revision is ${SVN_REVISION}.")
  add_definitions("-DHPX_SVN_REVISION=\"${SVN_REVISION}\"")
endif()

