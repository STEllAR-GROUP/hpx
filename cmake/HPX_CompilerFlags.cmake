# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_COMPILERFLAGS_LOADED TRUE)

include(HPX_Include)

hpx_include(Message
            GCCVersion
            AddConfigTest
            ParseArguments)

macro(hpx_use_flag_if_available flag)
  hpx_parse_arguments(FLAG "NAME" "" ${ARGN})

  set(uppercase_name "")

  if(FLAG_NAME)
    string(TOUPPER ${FLAG_NAME} uppercase_name)
  else()
    string(TOUPPER ${flag} uppercase_name)
  endif()

  add_hpx_config_test(${uppercase_name} HPX_FLAG_${uppercase_name} LANGUAGE CXX 
    SOURCE cmake/tests/flag.cpp
    FLAGS "-${flag}" FILE)

  if(HPX_FLAG_${uppercase_name})
    add_definitions("-${flag}")
  else()
    hpx_warn("use_flag_if_available" "${flag} is unavailable") 
  endif()
endmacro()

macro(hpx_use_flag_if_gcc_version flag version)
  if(${version} GREATER ${GCC_VERSION})
    hpx_warn("use_flag_if_gcc_version" "${flag} is unavailable") 
  else()
    add_definitions("-${flag}")
  endif()
endmacro()

macro(hpx_use_flag_if_msvc_version flag version)
  if(${version} GREATER ${MSVC_VERSION})
    hpx_warn("use_flag_if_msvc_version" "${flag} is unavailable") 
  else()
    add_definitions("-${flag}")
  endif()
endmacro()

