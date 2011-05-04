# Copyright (c) 2007-2009 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################
set(sources
    amr_c.cpp
    logging.cpp
    stencil.cpp)

set(headers
    logging.hpp
    stencil.hpp
    stencil_data.hpp
    stencil_data_locking.hpp
    stencil_functions.hpp)

###############################################################################
set(ini_files
    had_amr_c.ini)

###############################################################################
source_group("Source Files"        FILES ${sources})
source_group("Header Files"        FILES ${headers})
source_group("Configuration Files" FILES ${ini_files})

###############################################################################
add_hpx_component(had_amr_c
  MODULE had_amr
  SOURCES ${sources} 
  HEADERS ${headers}
  INI ${ini_files}
  DEPENDENCIES had_amr_c_test_component)

