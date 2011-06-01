################################################################################
#  Copyright (c) 2011 Bryce Lelbach 
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying 
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
################################################################################

include_directories(${PHXPR_INCLUDE_DIR})

add_hpx_executable(hpxi 
  MODULE hpxi
  SOURCES hpxi.cpp 
  DEPENDENCIES ${PHXPR_LIBRARY})

# add a dependency to the pseudo-target  
add_hpx_pseudo_dependencies(examples.hpxi hpxi_exe) 

