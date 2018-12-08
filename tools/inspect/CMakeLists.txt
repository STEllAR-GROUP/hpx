# Copyright (c) 2015 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# add inpect executable

add_hpx_executable(inspect
  AUTOGLOB
  NOLIBS
  FOLDER "Tools/Inspect")

# Set the basic search paths for the generated HPX headers
target_include_directories(inspect PRIVATE ${CMAKE_BINARY_DIR})

if((NOT MSVC) OR HPX_WITH_VCPKG)
  target_link_libraries(inspect PRIVATE ${HPX_TLL_PUBLIC} hpx::boost)
endif()

# add dependencies to pseudo-target
add_hpx_pseudo_dependencies(tools.inspect inspect)

