# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(OCLM_MESSAGE_LOADED TRUE)

macro(oclm_info type)
  string(TOLOWER ${type} lctype)
  message("[oclm.info.${lctype}] " ${ARGN})
endmacro()

macro(oclm_debug type)
  if("${OCLM_CMAKE_LOGLEVEL}" MATCHES "DEBUG|debug|Debug")
    string(TOLOWER ${type} lctype)
    message("[oclm.debug.${lctype}] " ${ARGN})
  endif()
endmacro()

macro(oclm_warn type)
  if("${OCLM_CMAKE_LOGLEVEL}" MATCHES "DEBUG|debug|Debug|WARN|warn|Warn")
    string(TOLOWER ${type} lctype)
    message("[oclm.warn.${lctype}] " ${ARGN})
  endif()
endmacro()

macro(oclm_error type)
  string(TOLOWER ${type} lctype)
  message(FATAL_ERROR "[oclm.error.${lctype}] " ${ARGN})
endmacro()

macro(oclm_message level type)
  if("${level}" MATCHES "ERROR|error|Error")
    string(TOLOWER ${type} lctype)
    oclm_error(${lctype} ${ARGN})
  elseif("${level}" MATCHES "WARN|warn|Warn")
    string(TOLOWER ${type} lctype)
    oclm_warn(${lctype} ${ARGN})
  elseif("${level}" MATCHES "DEBUG|debug|Debug")
    string(TOLOWER ${type} lctype)
    oclm_debug(${lctype} ${ARGN})
  elseif("${level}" MATCHES "INFO|info|Info")
    string(TOLOWER ${type} lctype)
    oclm_info(${lctype} ${ARGN})
  else()
    oclm_error("message" "\"${level}\" is not an OCLM configuration logging level.")
  endif()
endmacro()

macro(oclm_config_loglevel level return)
  set(${return} FALSE)
  if(    "${OCLM_CMAKE_LOGLEVEL}" MATCHES "ERROR|error|Error"
     AND "${level}" MATCHES "ERROR|error|Error")
    set(${return} TRUE)
  elseif("${OCLM_CMAKE_LOGLEVEL}" MATCHES "WARN|warn|Warn"
     AND "${level}" MATCHES "WARN|warn|Warn")
    set(${return} TRUE)
  elseif("${OCLM_CMAKE_LOGLEVEL}" MATCHES "DEBUG|debug|Debug"
     AND "${level}" MATCHES "DEBUG|debug|Debug")
    set(${return} TRUE)
  elseif("${OCLM_CMAKE_LOGLEVEL}" MATCHES "INFO|info|Info"
     AND "${level}" MATCHES "INFO|info|Info")
    set(${return} TRUE)
  endif()
endmacro()

macro(oclm_print_list level type message list)
  oclm_config_loglevel(${level} printed)
  if(printed)
    if(${list})
      oclm_message(${level} ${type} "${message}: ")
      foreach(element ${${list}})
        message("    ${element}")
      endforeach()
    else()
      oclm_message(${level} ${type} "${message} is empty.")
    endif()
  endif()
endmacro()
