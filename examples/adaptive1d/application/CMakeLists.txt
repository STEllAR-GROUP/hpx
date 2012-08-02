# Copyright (c) 2009-2011 Matthew Anderson
# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################
set(sources
    application.cpp)

###############################################################################
source_group("Source Files" FILES ${sources})

if(RNPL_FOUND)
  # add RNPL as a dependency
  set(dependencies
      ${RNPL_LIBRARY})
endif()

###############################################################################
add_hpx_library(adaptive1d_application
  MODULE adaptive1d
  SOURCES ${sources}
  DEPENDENCIES ${dependencies}
  FOLDER "Examples/Applications/Adaptive1d")

