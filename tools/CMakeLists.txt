# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tools
    cpu_features)

set(cpu_features NOLIBS DEPENDENCIES ${BOOST_program_options_LIBRARY})

foreach(tool ${tools})
  add_hpx_executable(${tool} 
    SOURCES ${tool}.cpp ${${tool}}
    FOLDER "Tools/${tool}")
  add_hpx_pseudo_dependencies(tools ${tool}_exe)
endforeach()

