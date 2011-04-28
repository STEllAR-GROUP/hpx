# Copyright (c) 2011 Bryce Lelbach 
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tools
    cpu_features
    dump_static_component_types
    linux_page_stats)

set(cpu_features NOLIBS DEPENDENCIES ${BOOST_program_options_LIBRARY})
set(linux_page_stats NOLIBS)

foreach(tool ${tools})
  add_hpx_executable(${tool} SOURCES ${tool}.cpp ${${tool}})
  add_hpx_pseudo_dependencies(tools ${tool}_exe) 
endforeach()

