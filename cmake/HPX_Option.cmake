# Copyright (c) 2014 Thomas Heller
# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(CMakeParseArguments)

set(HPX_OPTION_CATEGORIES
  "Generic"
  "Build Targets"
  "Thread Manager"
  "AGAS"
  "Parcelport"
  "Profiling"
  "Debugging"
)

function(hpx_option option type description default)
  set(options ADVANCED)
  set(one_value_args CATEGORY)
  set(multi_value_args STRINGS)
  cmake_parse_arguments(HPX_OPTION "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(NOT DEFINED ${option})
    set(${option} ${default} CACHE ${type} "${description}")
    if(HPX_OPTION_ADVANCED)
      mark_as_advanced(${option})
    endif()
  else()
    set_property(CACHE "${option}" PROPERTY HELPSTRING "${description}")
    set_property(CACHE "${option}" PROPERTY TYPE "${type}")
  endif()

  if(HPX_OPTION_STRINGS)
    if("${type}" STREQUAL "STRING")
      set_property(CACHE "${option}" PROPERTY STRINGS "${HPX_OPTION_STRINGS}")
    else()
      message(FATAL_ERROR "hpx_option(): STRINGS can only be used if type is STRING !")
    endif()
  endif()

  set(_category "Generic")
  if(HPX_OPTION_CATEGORY)
    set(_category "${HPX_OPTION_CATEGORY}")
  endif()
  set(${option}Category ${_category} CACHE INTERNAL "")
endfunction()

