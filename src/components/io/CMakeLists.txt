# Copyright (c) 2014 Shuangyang Yang
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################
set(root "${hpx_SOURCE_DIR}/hpx/components/io")

if(HPX_DEFAULT_BUILD_TARGETS)
  add_hpx_component(local_file
    FOLDER "Core/Components"
    HEADER_ROOT ${ROOT}
    SOURCES local_file.cpp
    ESSENTIAL)
else()
  add_hpx_component(local_file
    FOLDER "Core/Components"
    HEADER_ROOT ${ROOT}
    SOURCES local_file.cpp
    )
endif()

add_hpx_pseudo_dependencies(components.io
                            local_file_component)

################################################################################

find_package(OrangeFS)

if(ORANGEFS_FOUND)
  # add OrangeFS header files directory
  include_directories(${ORANGEFS_INCLUDE_DIR})

  if(HPX_DEFAULT_BUILD_TARGETS)
    add_hpx_component(orangefs_file
      FOLDER "Core/Components"
      HEADER_ROOT ${ROOT}
      SOURCES orangefs_file.cpp
      DEPENDENCIES ${ORANGEFS_LIBRARY}
      ESSENTIAL)
  else()
    add_hpx_component(orangefs_file
      FOLDER "Core/Components"
      HEADER_ROOT ${ROOT}
      SOURCES orangefs_file.cpp
      DEPENDENCIES ${ORANGEFS_LIBRARY}
      )
  endif()

  add_hpx_pseudo_dependencies(components.io
                              orangefs_file_component)
endif()
