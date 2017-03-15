# Copyright (c) 2014 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tools)
set(subdirs inspect)


if(NOT MSVC)
  set(tools ${tools} cpu_features)
  set(cpu_features NOLIBS DEPENDENCIES ${BOOST_program_options_LIBRARY})
endif()

foreach(tool ${tools})
  add_hpx_executable(${tool}
    SOURCES ${tool}.cpp ${${tool}}
    FOLDER "Tools/${tool}")
  add_hpx_pseudo_dependencies(tools ${tool}_exe)
  if(NOT HPX_WITH_CUDA)  # Cuda needs plain target_link_libraries() signature
    target_link_libraries(${tool}_exe ${Boost_LIBRARIES})
  else()
    target_link_libraries(${tool}_exe PUBLIC ${Boost_LIBRARIES})
  endif()
endforeach()

foreach(subdir ${subdirs})
  add_hpx_pseudo_target(tools.${subdir})
  add_subdirectory(${subdir})
  add_hpx_pseudo_dependencies(tools tools.${subdir})
endforeach()

