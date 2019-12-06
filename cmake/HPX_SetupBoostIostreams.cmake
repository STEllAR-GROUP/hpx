# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_COMPRESSION_BZIP2 OR HPX_WITH_COMPRESSION_ZLIB)

  find_package(Boost ${Boost_MINIMUM_VERSION} QUIET MODULE COMPONENTS iostreams)

  if(Boost_IOSTREAMS_FOUND)
    hpx_info("  iostreams")
  else()
    hpx_error("Could not find Boost.Iostreams but HPX_WITH_COMPRESSION_BZIP2=On or \
    HPX_WITH_COMPRESSION_LIB=On. Either set it to off or provide a boost installation including \
    the iostreams library")
  endif()

  add_library(hpx::boost::iostreams INTERFACE IMPORTED)

  # Can't directly link to "iostreams" target in set_property, can change is when using target_link_libraries

  set_property(TARGET hpx::boost::iostreams APPEND PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIRS})
  if(${CMAKE_VERSION} VERSION_LESS "3.12.0")
    set_property(TARGET hpx::boost::iostreams APPEND PROPERTY
      INTERFACE_LINK_LIBRARIES ${Boost_IOSTREAMS_LIBRARIES})
  else()
    target_link_libraries(hpx::boost::iostreams INTERFACE
      ${Boost_IOSTREAMS_LIBRARIES})
  endif()

endif()
