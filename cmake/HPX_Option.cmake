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
  "Modules"
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

# simplify setting an option in cache
function(hpx_set_option option)
  set(options FORCE)
  set(one_value_args VALUE TYPE HELPSTRING)
  set(multi_value_args)
  cmake_parse_arguments(HPX_SET_OPTION "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(NOT DEFINED ${option})
    hpx_error("attempting to set an undefined option: ${option}")
  endif()

  set(${option}_force)
  if(HPX_SET_OPTION_FORCE)
    set(${option}_force FORCE)
  endif()

  if(HPX_SET_OPTION_HELPSTRING)
    set(${option}_description ${HPX_SET_OPTION_HELPSTRING})
  else()
    get_property(${option}_description CACHE "${option}" PROPERTY HELPSTRING)
  endif()

  if(HPX_SET_OPTION_TYPE)
    set(${option}_type ${HPX_SET_OPTION_TYPE})
  else()
    get_property(${option}_type CACHE "${option}" PROPERTY TYPE)
  endif()

  if(DEFINED HPX_SET_OPTION_VALUE)
    set(${option}_value ${HPX_SET_OPTION_VALUE})
  else()
    get_property(${option}_value CACHE "${option}" PROPERTY VALUE)
  endif()

  set(${option} ${${option}_value} CACHE ${${option}_type} "${${option}_description}" ${${option}_force})
endfunction()
