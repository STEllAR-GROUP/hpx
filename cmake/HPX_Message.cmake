# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_MESSAGE_LOADED TRUE)

macro(hpx_info type)
  string(TOLOWER ${type} lctype)
  message("[hpx.info.${lctype}] " ${ARGN})
endmacro()

macro(hpx_debug type)
  if("${HPX_CMAKE_LOGLEVEL}" MATCHES "DEBUG|debug|Debug")
    string(TOLOWER ${type} lctype)
    message("[hpx.debug.${lctype}] " ${ARGN})
  endif()
endmacro()

macro(hpx_warn type)
  if("${HPX_CMAKE_LOGLEVEL}" MATCHES "DEBUG|debug|Debug|WARN|warn|Warn")
    string(TOLOWER ${type} lctype)
    message("[hpx.warn.${lctype}] " ${ARGN})
  endif()
endmacro()

macro(hpx_error type)
  string(TOLOWER ${type} lctype)
  message(FATAL_ERROR "[hpx.error.${lctype}] " ${ARGN})
endmacro()

macro(hpx_message level type)
  if("${level}" MATCHES "ERROR|error|Error")
    string(TOLOWER ${type} lctype)
    hpx_error(${lctype} ${ARGN})
  elseif("${level}" MATCHES "WARN|warn|Warn")
    string(TOLOWER ${type} lctype)
    hpx_warn(${lctype} ${ARGN})
  elseif("${level}" MATCHES "DEBUG|debug|Debug")
    string(TOLOWER ${type} lctype)
    hpx_debug(${lctype} ${ARGN})
  elseif("${level}" MATCHES "INFO|info|Info")
    string(TOLOWER ${type} lctype)
    hpx_info(${lctype} ${ARGN})
  else()
    hpx_error("message" "\"${level}\" is not an HPX configuration logging level.")
  endif()
endmacro()

macro(hpx_config_loglevel level return)
  set(${return} FALSE)
  if(    "${HPX_CMAKE_LOGLEVEL}" MATCHES "ERROR|error|Error"
     AND "${level}" MATCHES "ERROR|error|Error")
    set(${return} TRUE)
  elseif("${HPX_CMAKE_LOGLEVEL}" MATCHES "WARN|warn|Warn"
     AND "${level}" MATCHES "WARN|warn|Warn")
    set(${return} TRUE)
  elseif("${HPX_CMAKE_LOGLEVEL}" MATCHES "DEBUG|debug|Debug"
     AND "${level}" MATCHES "DEBUG|debug|Debug")
    set(${return} TRUE)
  elseif("${HPX_CMAKE_LOGLEVEL}" MATCHES "INFO|info|Info"
     AND "${level}" MATCHES "INFO|info|Info")
    set(${return} TRUE)
  endif()
endmacro()

macro(hpx_print_list level type message list)
  hpx_config_loglevel(${level} printed)
  if(printed)
    if(${list})
      hpx_message(${level} ${type} "${message}: ")
      foreach(element ${${list}})
        message("    ${element}")
      endforeach()
    else()
      hpx_message(${level} ${type} "${message} is empty.")
    endif()
  endif()
endmacro()

