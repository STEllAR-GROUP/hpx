# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c)      2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(root "${hpx_SOURCE_DIR}/hpx/components/iostreams")

add_hpx_library_headers(iostreams
  GLOB_RECURSE GLOBS "${root}/*.h*")

add_hpx_library_sources(iostreams
  GLOB_RECURSE GLOBS "*.c*")

add_definitions(-DHPX_IOSTREAMS_MODULE_EXPORTS)

add_hpx_component(iostreams
  SOURCES ${iostreams_SOURCES}
  HEADERS ${iostreams_HEADERS}
  FOLDER "Core"
  HEADER_ROOT ${root}
  SOURCE_ROOT ".")

