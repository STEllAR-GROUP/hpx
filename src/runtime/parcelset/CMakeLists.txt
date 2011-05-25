# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_hpx_library_sources(hpx
  parcel.cpp
  parcelport.cpp
  parcelport_connection.cpp)

if("${HPX_AGAS_VERSION}" STREQUAL "2")
  add_hpx_library_sources(hpx parcelhandler_v2.cpp)
else()
  add_hpx_library_sources(hpx parcelhandler_v1.cpp)
endif()

add_subdirectory(server)

