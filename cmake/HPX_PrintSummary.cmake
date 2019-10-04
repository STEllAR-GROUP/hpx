# Copyright (c) 2017-2019 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

function(create_configuration_summary message module_name)

  hpx_info("")
  hpx_info(${message})

  set(hpx_config_information)
  set(upper_cfg_name "HPX")

  string(TOUPPER ${module_name} module_name_uc)
  if(NOT "${module_name_uc}x" STREQUAL "HPXx")
    set(upper_cfg_name "HPX_${module_name_uc}")
  endif()

  get_property(DEFINITIONS_VARS GLOBAL PROPERTY HPX_CONFIG_DEFINITIONS)
  if(DEFINED DEFINITIONS_VARS)
    list(SORT DEFINITIONS_VARS)
    list(REMOVE_DUPLICATES DEFINITIONS_VARS)
  endif()

  get_cmake_property(_variableNames CACHE_VARIABLES)
  foreach (_variableName ${_variableNames})
    if(${_variableName}Category)

      # handle only options which start with HPX_WITH_
      string(FIND ${_variableName} "${upper_cfg_name}_WITH_" __pos)

#      hpx_info("  ${_variableName} ${module_name} ${module_name_uc}, ${upper_cfg_name} ${__pos}")

      if(${__pos} EQUAL 0)
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

  if("${module_name}x" STREQUAL "hpxx")
    set(_base_dir_local "hpx/config")
    set(_base_dir "hpx/config")
    set(_template "config_defines_strings.hpp.in")
  else()
    set(_base_dir_local "libs/${module_name}/include/hpx/${module_name}/config/")
    set(_base_dir "hpx/${module_name}/config/")
    set(_template "config_defines_strings_for_modules.hpp.in")
  endif()

  configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/templates/${_template}"
    "${CMAKE_BINARY_DIR}/${_base_dir_local}/config_strings.hpp"
    @ONLY)
  configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/templates/${_template}"
    "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${_base_dir}/config_strings.hpp"
    @ONLY)

endfunction()

