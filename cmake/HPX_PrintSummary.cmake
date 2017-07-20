# Copyright (c) 2017 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

get_property(DEFINITIONS_VARS GLOBAL PROPERTY HPX_CONFIG_DEFINITIONS)
if(DEFINED DEFINITIONS_VARS)
  list(SORT DEFINITIONS_VARS)
  list(REMOVE_DUPLICATES DEFINITIONS_VARS)
endif()

set(hpx_config_information)

message("")
hpx_info("Configuration summary:")
get_cmake_property(_variableNames CACHE_VARIABLES)
foreach (_variableName ${_variableNames})
  if(${_variableName}Category)

    # handle only opetions which start with HPX_WITH_
    string(FIND ${_variableName} "HPX_WITH_" __pos)
    if(NOT ${__pos} EQUAL -1)
      get_property(_value CACHE "${_variableName}" PROPERTY VALUE)
      hpx_info("  ${_variableName}=${_value}")

      string(REPLACE "_WITH_" "_HAVE_" __variableName ${_variableName})
      list(FIND DEFINITIONS_VARS ${__variableName} __pos)
      if(NOT ${__pos} EQUAL -1)
        set(hpx_config_information
            "${hpx_config_information}"
            "\n        \"${_variableName}=${_value}\",")
      elseif(NOT ${_variableName}Category STREQUAL "Generic" AND NOT ${_variableName}Category STREQUAL "Build Targets")
        get_property(_type CACHE "${_variableName}" PROPERTY TYPE)
        if(_type STREQUAL "BOOL")
          set(hpx_config_information
              "${hpx_config_information}"
              "\n        \"${_variableName}=OFF\",")
        endif()
      endif()

    endif()
  endif()
endforeach()

if(hpx_config_information)
  string(REPLACE ";" "" hpx_config_information ${hpx_config_information})
endif()
configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/templates/config_defines_strings.hpp.in"
  "${CMAKE_BINARY_DIR}/hpx/config/config_strings.hpp"
  @ONLY)
configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/templates/config_defines_strings.hpp.in"
  "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/hpx/config/config_strings.hpp"
  @ONLY)
