# Copyright (c) 2011      Bryce Lelbach
# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c) 2007-2011 Matthew Anderson
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################
set(sources
    stencil_register.cpp
    logging.cpp
    stencil.cpp)

set(headers
    logging.hpp
    stencil.hpp
    stencil_data.hpp
    stencil_data_locking.hpp
    stencil_functions.hpp)

###############################################################################
source_group("Source Files" FILES ${sources})
source_group("Header Files" FILES ${headers})

###############################################################################
add_hpx_component(adaptive1d_stencil
  MODULE adaptive1d
  SOURCES ${sources}
  HEADERS ${headers}
  DEPENDENCIES adaptive1d_application_lib
  FOLDER "Examples/Adaptive1d")

