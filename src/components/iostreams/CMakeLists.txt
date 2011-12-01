# Copyright (c) 2007-2011 Hartmut Kaiser
# Copyright (c)      2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_definitions(-DHPX_IOSTREAMS_MODULE_EXPORTS)

add_hpx_component(iostreams
  SOURCES component_module.cpp
          manipulators.cpp
          standard_streams.cpp
          server/output_stream.cpp)

