# Copyright (c) 2019 The STE||AR-Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(MSVC)
  # cmake will not create a separate VS project without any source files, thus
  # this adds a dummy (empty) source file to the target
  set(sources ${CMAKE_CURRENT_SOURCE_DIR}/pp.cpp)

  target_sources(hpx_preprocessor PRIVATE ${sources})

  add_hpx_source_group(
    NAME hpx_preprocessor
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    CLASS "Source Files"
    TARGETS ${sources})
endif()
