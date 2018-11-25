# Copyright (c) 2014-2017 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tools)
set(subdirs inspect)


if((NOT MSVC) OR HPX_WITH_VCPKG)
  add_hpx_executable(cpu_features
    SOURCES cpu_features.cpp
    NOLIBS
    FOLDER "Tools/cpu_features")

  add_hpx_pseudo_dependencies(tools cpu_features)
  target_link_libraries(cpu_features PRIVATE ${HPX_TLL_PUBLIC} Boost::program_options)
endif()

foreach(subdir ${subdirs})
  add_hpx_pseudo_target(tools.${subdir})
  add_subdirectory(${subdir})
  add_hpx_pseudo_dependencies(tools tools.${subdir})
endforeach()

