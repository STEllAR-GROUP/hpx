# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c)      2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################
set(root "${hpx_SOURCE_DIR}/hpx/components/distributing_factory")

add_hpx_library_headers(distributing_factory
  GLOB_RECURSE GLOBS "${root}/*.h*")

add_hpx_library_sources(distributing_factory
  GLOB_RECURSE GLOBS "*.c*")

###############################################################################
add_hpx_component(distributing_factory
  SOURCES ${distributing_factory_SOURCES}
  HEADERS ${distributing_factory_HEADERS}
  FOLDER "Core"
  HEADER_ROOT ${root}
  SOURCE_ROOT ".")

