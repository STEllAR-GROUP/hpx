# Copyright (c) 2014-2017 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tools)
set(subdirs inspect)

set(tools ${tools} cpu_features)
set(cpu_features NOLIBS DEPENDENCIES ${BOOST_program_options_LIBRARY}
  hpx_hardware hpx_program_options)

foreach(tool ${tools})
  add_hpx_executable(${tool}
    SOURCES ${tool}.cpp ${${tool}}
    FOLDER "Tools/${tool}")
  add_hpx_pseudo_dependencies(tools ${tool})
  target_link_libraries(${tool} PRIVATE ${Boost_LIBRARIES}
    hpx_config hpx_hardware hpx_program_options)
  target_compile_definitions(${tool} PRIVATE HPX_MODULE_STATIC_LINKING)
endforeach()

foreach(subdir ${subdirs})
  add_hpx_pseudo_target(tools.${subdir})
  add_subdirectory(${subdir})
  add_hpx_pseudo_dependencies(tools tools.${subdir})
endforeach()

