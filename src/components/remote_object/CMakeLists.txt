# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c)      2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################
set(root "${hpx_SOURCE_DIR}/hpx/components/remote_object")

add_hpx_library_headers(remote_object
  GLOB_RECURSE GLOBS "${root}/*.h*")

add_hpx_library_sources(remote_object
  GLOB_RECURSE GLOBS "*.c*")

###############################################################################
add_hpx_component(remote_object
  SOURCES ${remote_object_SOURCES}
  HEADERS ${remote_object_HEADERS}
  FOLDER "Core"
  HEADER_ROOT ${root}
  SOURCE_ROOT ".")

