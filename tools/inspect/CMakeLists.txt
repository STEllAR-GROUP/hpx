# Copyright (c) 2015 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# add inpect executable

add_hpx_executable(inspect
  AUTOGLOB
  NOLIBS
  FOLDER "Tools/Inspect")

if(NOT Boost_REGEX_FOUND)
  hpx_error("HPX inspect tool requires Boost.Regex")
endif()

target_link_libraries(inspect PRIVATE ${Boost_LIBRARIES} 
  hpx_config hpx_filesystem hpx_program_options)
target_compile_definitions(inspect PRIVATE HPX_MODULE_STATIC_LINKING)

# add dependencies to pseudo-target
add_hpx_pseudo_dependencies(tools.inspect inspect)

