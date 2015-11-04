# Copyright (c) 2014 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(CMakeParseArguments)

macro(hpx_add_compile_flag FLAG)
  set(options)
  set(one_value_args)
  set(multi_value_args CONFIGURATIONS LANGUAGES)
  cmake_parse_arguments(HPX_ADD_COMPILE_FLAG "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  set(_languages "CXX")
  if(HPX_ADD_COMPILE_FLAG_LANGUAGES)
    set(_languages ${HPX_ADD_COMPILE_FLAG_LANGUAGES})
  endif()

  set(_configurations "none")
  if(HPX_ADD_COMPILE_FLAG_CONFIGURATIONS)
    set(_configurations "${HPX_ADD_COMPILE_FLAG_CONFIGURATIONS}")
  endif()

  foreach(_lang ${_languages})
    foreach(_config ${_configurations})
      set(_conf)
      if(NOT _config STREQUAL "none")
        string(TOUPPER "${_config}" _conf)
        set(_conf "_${_conf}")
      endif()
      set(CMAKE_${_lang}_FLAGS${_conf} "${CMAKE_${_lang}_FLAGS${_conf}} ${FLAG}")
    endforeach()
  endforeach()
endmacro()

macro(hpx_remove_compile_flag FLAG)
  set(options)
  set(one_value_args)
  set(multi_value_args CONFIGURATIONS LANGUAGES)
  cmake_parse_arguments(HPX_REMOVE_COMPILE_FLAG "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  set(_languages "CXX")
  if(HPX_REMOVE_COMPILE_FLAG_LANGUAGES)
    set(_languages ${HPX_REMOVE_COMPILE_FLAG_LANGUAGES})
  endif()

  set(_configurations "none")
  if(HPX_REMOVE_COMPILE_FLAG_CONFIGURATIONS)
    set(_configurations "${HPX_REMOVE_COMPILE_FLAG_CONFIGURATIONS}")
  endif()

  foreach(_lang ${_languages})
    foreach(_config ${_configurations})
      set(_conf)
      if(NOT _config STREQUAL "none")
        string(TOUPPER "${_config}" _conf)
        set(_conf "_${_conf}")
      endif()
      STRING (REGEX REPLACE "${FLAG}" "" CMAKE_${_lang}_FLAGS${_conf} "${CMAKE_${_lang}_FLAGS${_conf}}")
    endforeach()
  endforeach()
endmacro()


include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)

macro(hpx_add_compile_flag_if_available FLAG)
  set(options)
  set(one_value_args NAME)
  set(multi_value_args CONFIGURATIONS LANGUAGES)
  cmake_parse_arguments(HPX_ADD_COMPILE_FLAG_IA "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(HPX_ADD_COMPILE_FLAG_IA_NAME)
    string(TOUPPER ${HPX_ADD_COMPILE_FLAG_IA_NAME} _name)
  else()
    string(TOUPPER ${FLAG} _name)
  endif()

  string(REGEX REPLACE "^-+" "" _name ${_name})
  string(REGEX REPLACE "[=\\-]" "_" _name ${_name})
  string(REGEX REPLACE "," "_" _name ${_name})
  string(REGEX REPLACE "\\+" "X" _name ${_name})

  set(_languages "CXX")
  if(HPX_ADD_COMPILE_FLAG_IA_LANGUAGES)
    set(_languages ${HPX_ADD_COMPILE_FLAG_IA_LANGUAGES})
  endif()

  foreach(_lang ${_languages})
    if(_lang STREQUAL "CXX")
      check_cxx_compiler_flag(${FLAG} HPX_WITH_${_lang}_FLAG_${_name})
    elseif(_lang STREQUAL "C")
      check_c_compiler_flag(${FLAG} HPX_WITH_${_lang}_FLAG_${_name})
    elseif(_lang STREQUAL "Fortran")
      ## Assuming the C compiler accepts the same flags as the fortran one ...
      check_c_compiler_flag(${FLAG} HPX_WITH_${_lang}_FLAG_${_name})
    else()
      hpx_error("Unsupported language ${_lang}.")
    endif()
    if(HPX_WITH_${_lang}_FLAG_${_name})
      hpx_add_compile_flag(${FLAG} CONFIGURATIONS ${HPX_ADD_COMPILE_FLAG_IA_CONFIGURATIONS} LANGUAGES ${_lang})
    else()
      hpx_info("\"${FLAG}\" not available for language ${_lang}.")
    endif()
  endforeach()

endmacro()
