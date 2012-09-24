# Copyright (c) 2007-2011 Hartmut Kaiser
# Copyright (c)      2011 Bryce Lelbach
# Copyright (c)      2012 Maciej Brodowicz
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(root "${hpx_SOURCE_DIR}/hpx/components/papi")

add_hpx_library_headers(papi_counters
  GLOB_RECURSE GLOBS "${root}/*.hpp")

add_hpx_library_sources(papi_counters
  GLOB_RECURSE GLOBS "*.cpp")

add_hpx_component(papi_counters
  SOURCES ${papi_counters_SOURCES}
  HEADERS ${papi_counters_HEADERS}
  DEPENDENCIES ${PAPI_LIBRARY}
  FOLDER "Core"
  HEADER_ROOT ${root}
  SOURCE_ROOT ".")
