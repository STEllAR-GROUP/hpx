# Copyright (c) 2015 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# add inpect executable

include(HPX_SetupBoostRegex)

add_hpx_executable(
  inspect INTERNAL_FLAGS AUTOGLOB NOLIBS FOLDER "Tools/Inspect"
)

if(NOT Boost_REGEX_FOUND)
  hpx_error("HPX inspect tool requires Boost.Regex")
endif()

# Set the basic search paths for the generated HPX headers
target_include_directories(inspect PRIVATE ${PROJECT_BINARY_DIR})
target_link_libraries(
  inspect PRIVATE Boost::regex hpx_dependencies_boost hpx_core
)

# add dependencies to pseudo-target
add_hpx_pseudo_dependencies(tools.inspect inspect)
