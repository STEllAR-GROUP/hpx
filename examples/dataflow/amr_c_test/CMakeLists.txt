# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c) 2009-2011 Matthew Anderson
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################
set(sources
    dataflow_c_test.cpp)

###############################################################################
source_group("Source Files" FILES ${sources})

###############################################################################
add_hpx_library(dataflow_c_test
  MODULE dataflow_c_test
  SOURCES ${sources}
  FOLDER "Examples/Applications/AMR")

