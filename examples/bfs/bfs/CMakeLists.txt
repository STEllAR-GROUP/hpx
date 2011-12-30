# Copyright (c) 2011 Matt Anderson
# Copyright (c) 2007-2012 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(root "${hpx_SOURCE_DIR}/examples/bfs/bfs")

add_hpx_library_headers(bfs
  GLOB_RECURSE GLOBS "${root}/*.h*")

add_hpx_library_sources(bfs
  GLOB_RECURSE GLOBS "*.c*")

add_definitions(-DHPX_IOSTREAMS_MODULE_EXPORTS)

add_hpx_component(bfs
  MODULE bfs
  SOURCES ${bfs_SOURCES}
  HEADERS ${bfs_HEADERS}
  DEPENDENCIES iostreams_component
  FOLDER "Examples/Bfs"
  HEADER_ROOT ${root}
  SOURCE_ROOT ".")

