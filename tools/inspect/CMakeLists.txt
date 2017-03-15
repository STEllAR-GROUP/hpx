# Copyright (c) 2015 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# add inpect executable

add_hpx_executable(inspect
  AUTOGLOB
  NOLIBS
  FOLDER "Tools/Inspect")

if(NOT MSVC)
  if(NOT HPX_WITH_CUDA)  # Cuda needs plain target_link_libraries() signature
    target_link_libraries(inspect_exe ${Boost_LIBRARIES})
  else()
    target_link_libraries(inspect_exe PUBLIC ${Boost_LIBRARIES})
  endif()
endif()

# add dependencies to pseudo-target
add_hpx_pseudo_dependencies(tools.inspect inspect_exe)

