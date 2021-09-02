# Copyright (c) 2011 Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

function(hpx_local_to_string var)
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

function(hpx_local_info)
  set(msg)
  hpx_local_to_string(msg ${ARGN})
  message(STATUS "${msg}")
  unset(args)
endfunction()

function(hpx_local_debug)
  if("${HPXLocal_CMAKE_LOGLEVEL}" MATCHES "DEBUG|debug|Debug")
    set(msg "DEBUG:")
    hpx_local_to_string(msg ${ARGN})
    message(STATUS "${msg}")
  endif()
endfunction()

function(hpx_local_warn)
  if("${HPXLocal_CMAKE_LOGLEVEL}" MATCHES "DEBUG|debug|Debug|WARN|warn|Warn")
    set(msg "WARNING:")
    hpx_local_to_string(msg ${ARGN})
    message(STATUS "${msg}")
  endif()
endfunction()

function(hpx_local_error)
  set(msg "ERROR:")
  hpx_local_to_string(msg ${ARGN})
  message(FATAL_ERROR "${msg}")
endfunction()

function(hpx_local_message level)
  if("${level}" MATCHES "ERROR|error|Error")
    hpx_local_error(${ARGN})
  elseif("${level}" MATCHES "WARN|warn|Warn")
    hpx_local_warn(${ARGN})
  elseif("${level}" MATCHES "DEBUG|debug|Debug")
    hpx_local_debug(${ARGN})
  elseif("${level}" MATCHES "INFO|info|Info")
    hpx_local_info(${ARGN})
  else()
    hpx_local_error(
      "message" "\"${level}\" is not an HPX configuration logging level."
    )
  endif()
endfunction()

function(hpx_local_config_loglevel level return)
  set(${return}
      FALSE
      PARENT_SCOPE
  )
  if("${HPXLocal_CMAKE_LOGLEVEL}" MATCHES "ERROR|error|Error"
     AND "${level}" MATCHES "ERROR|error|Error"
  )
    set(${return}
        TRUE
        PARENT_SCOPE
    )
  elseif("${HPXLocal_CMAKE_LOGLEVEL}" MATCHES "WARN|warn|Warn"
         AND "${level}" MATCHES "WARN|warn|Warn"
  )
    set(${return}
        TRUE
        PARENT_SCOPE
    )
  elseif("${HPXLocal_CMAKE_LOGLEVEL}" MATCHES "DEBUG|debug|Debug"
         AND "${level}" MATCHES "DEBUG|debug|Debug"
  )
    set(${return}
        TRUE
        PARENT_SCOPE
    )
  elseif("${HPXLocal_CMAKE_LOGLEVEL}" MATCHES "INFO|info|Info"
         AND "${level}" MATCHES "INFO|info|Info"
  )
    set(${return}
        TRUE
        PARENT_SCOPE
    )
  endif()
endfunction()

function(hpx_local_print_list level message list)
  hpx_local_config_loglevel(${level} printed)
  if(printed)
    if(${list})
      hpx_local_message(${level} "${message}: ")
      foreach(element ${${list}})
        message("    ${element}")
      endforeach()
    else()
      hpx_local_message(${level} "${message} is empty.")
    endif()
  endif()
endfunction()
