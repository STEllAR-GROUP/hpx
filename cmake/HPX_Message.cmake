# Copyright (c) 2011 Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

function(_to_string var)
  set(_var "")

  foreach(_arg ${ARGN})
    string(REPLACE "\\" "/" _arg ${_arg})
    if("${_var}" STREQUAL "")
      set(_var "${_arg}")
    else()
      set(_var "${_var} ${_arg}")
    endif()
  endforeach()

  set(${var}
      ${_var}
      PARENT_SCOPE
  )
endfunction()

function(hpx_info)
  set(msg)
  _to_string(msg ${ARGN})
  message(STATUS "${msg}")
  unset(args)
endfunction()

function(hpx_debug)
  if("${HPX_CMAKE_LOGLEVEL}" MATCHES "DEBUG|debug|Debug")
    set(msg)
    _to_string(msg ${ARGN})
    message(STATUS "DEBUG: ${msg}")
  endif()
endfunction()

function(hpx_warn)
  if("${HPX_CMAKE_LOGLEVEL}" MATCHES "DEBUG|debug|Debug|WARN|warn|Warn")
    set(msg)
    _to_string(msg ${ARGN})
    message(STATUS "WARNING: ${msg}")
  endif()
endfunction()

function(hpx_error)
  set(msg)
  _to_string(msg ${ARGN})
  message(FATAL_ERROR "ERROR: ${msg}")
endfunction()

function(hpx_message level)
  if("${level}" MATCHES "ERROR|error|Error")
    hpx_error(${ARGN})
  elseif("${level}" MATCHES "WARN|warn|Warn")
    hpx_warn(${ARGN})
  elseif("${level}" MATCHES "DEBUG|debug|Debug")
    hpx_debug(${ARGN})
  elseif("${level}" MATCHES "INFO|info|Info")
    hpx_info(${ARGN})
  else()
    hpx_error("message"
              "\"${level}\" is not an HPX configuration logging level."
    )
  endif()
endfunction()

function(hpx_config_loglevel level return)
  set(${return}
      FALSE
      PARENT_SCOPE
  )
  if("${HPX_CMAKE_LOGLEVEL}" MATCHES "ERROR|error|Error"
     AND "${level}" MATCHES "ERROR|error|Error"
  )
    set(${return}
        TRUE
        PARENT_SCOPE
    )
  elseif("${HPX_CMAKE_LOGLEVEL}" MATCHES "WARN|warn|Warn"
         AND "${level}" MATCHES "WARN|warn|Warn"
  )
    set(${return}
        TRUE
        PARENT_SCOPE
    )
  elseif("${HPX_CMAKE_LOGLEVEL}" MATCHES "DEBUG|debug|Debug"
         AND "${level}" MATCHES "DEBUG|debug|Debug"
  )
    set(${return}
        TRUE
        PARENT_SCOPE
    )
  elseif("${HPX_CMAKE_LOGLEVEL}" MATCHES "INFO|info|Info"
         AND "${level}" MATCHES "INFO|info|Info"
  )
    set(${return}
        TRUE
        PARENT_SCOPE
    )
  endif()
endfunction()

function(hpx_print_list level message list)
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
endfunction()
