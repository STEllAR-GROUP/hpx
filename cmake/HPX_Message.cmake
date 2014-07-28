# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

macro(hpx_info type)
  message(STATUS "${type} " ${ARGN})
endmacro()

macro(hpx_debug type)
  if("${HPX_CMAKE_LOGLEVEL}" MATCHES "DEBUG|debug|Debug")
    message("DEBUG: ${type} " ${ARGN})
  endif()
endmacro()

macro(hpx_warn type)
  if("${HPX_CMAKE_LOGLEVEL}" MATCHES "DEBUG|debug|Debug|WARN|warn|Warn")
    message("WARNING: ${type} " ${ARGN})
  endif()
endmacro()

macro(hpx_error type)
  message(FATAL_ERROR "ERROR: ${type} " ${ARGN})
endmacro()

macro(hpx_message level)
  if("${level}" MATCHES "ERROR|error|Error")
    hpx_error(${ARGN})
  elseif("${level}" MATCHES "WARN|warn|Warn")
    hpx_warn(${ARGN})
  elseif("${level}" MATCHES "DEBUG|debug|Debug")
    hpx_debug(${ARGN})
  elseif("${level}" MATCHES "INFO|info|Info")
    hpx_info(${ARGN})
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

macro(hpx_print_list level message list)
  hpx_config_loglevel(${level} printed)
  if(printed)
    if(${list})
      hpx_message(${level} "${message}: ")
      foreach(element ${${list}})
        message("    ${element}")
      endforeach()
    else()
      hpx_message(${level} "${message} is empty.")
    endif()
  endif()
endmacro()

