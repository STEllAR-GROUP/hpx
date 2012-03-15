# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_TARGETPATHS_LOADED TRUE)

macro(hpx_get_target_location variable target)
  if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(location LOCATION_DEBUG)
  elseif("${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")
    set(location LOCATION_RELWITHDEBINFO)
  elseif("${CMAKE_BUILD_TYPE}" STREQUAL "MinSizeRel")
    set(location LOCATION_MINSIZEREL)
  elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    set(location LOCATION_RELEASE)
  else()
    set(location LOCATION)
  endif()

  get_target_property(${variable} ${target} ${location})
endmacro()

macro(hpx_get_target_file variable target)
  hpx_get_target_location(location ${target})
  get_filename_component(${variable} "${location}" NAME)
endmacro()

macro(hpx_get_target_path variable target)
  hpx_get_target_location(location ${target})
  get_filename_component(${variable} "${location}" PATH)
endmacro()

