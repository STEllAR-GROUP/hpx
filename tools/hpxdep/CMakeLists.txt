# Copyright (c) 2015-2022 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# add hpxdep executable

add_hpx_executable(hpxdep INTERNAL_FLAGS AUTOGLOB NOLIBS FOLDER "Tools/HPXDep")

# Set the basic search paths for the generated HPX headers
target_include_directories(hpxdep PRIVATE ${PROJECT_BINARY_DIR})
target_link_libraries(hpxdep PRIVATE hpx_core)

# add dependencies to pseudo-target
add_hpx_pseudo_dependencies(tools.hpxdep hpxdep)

# generate dependency report
if(HPX_WITH_GIT_TAG)
  set(HPX_DEPREPORT_VERSION "${HPX_WITH_GIT_TAG}")
  set(HPX_DEPREPORT_CAPTION "Tag ${HPX_WITH_GIT_TAG}")
elseif(HPX_WITH_GIT_BRANCH)
  set(HPX_DEPREPORT_VERSION "${HPX_WITH_GIT_BRANCH}")
  set(HPX_DEPREPORT_CAPTION "Branch ${HPX_WITH_GIT_BRANCH}")
else()
  # HPX_VERSION is always available, use as fallback
  set(HPX_DEPREPORT_VERSION "latest")
  set(HPX_DEPREPORT_CAPTION "Version ${HPX_VERSION_FULL}")
endif()

# cmake-format: off
add_custom_target(depreport
  DEPENDS hpxdep
  COMMAND "${CMAKE_COMMAND}"
    -DHPX_DEPREPORT_VERSION=${HPX_DEPREPORT_VERSION}
    -DHPX_SOURCE_DIR:PATH="${PROJECT_SOURCE_DIR}"
    -DHPX_BINARY_DIR:PATH="${PROJECT_BINARY_DIR}"
    -DHPXDEP_OUTPUT_NAME:FILEPATH="$<TARGET_FILE:hpxdep>"
    -DHPX_CORE_ENABLED_MODULES="${HPX_CORE_ENABLED_MODULES}"
    -DHPX_FULL_ENABLED_MODULES="${HPX_FULL_ENABLED_MODULES}"
    -P "${PROJECT_SOURCE_DIR}/cmake/HPX_GenerateDependencyReport.cmake"
)
# cmake-format: on

set_target_properties(
  depreport PROPERTIES FOLDER "Documentation/Dependency Report"
)

if(TARGET git_docs)
  add_dependencies(git_docs depreport)
endif()
