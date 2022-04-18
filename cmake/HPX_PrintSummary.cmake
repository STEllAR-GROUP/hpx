# Copyright (c) 2017-2022 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

function(has_configuration_summary module_name has_config_info)

  string(TOUPPER ${module_name} __module_name_uc)
  get_property(
    _variableNames GLOBAL PROPERTY HPX_MODULE_CONFIG_${__module_name_uc}
  )
  list(LENGTH _variableNames _length)
  if(${_length} GREATER_EQUAL 1)
    set(${has_config_info}
        TRUE
        PARENT_SCOPE
    )
  else()
    set(${has_config_info}
        FALSE
        PARENT_SCOPE
    )
  endif()

endfunction()

function(create_configuration_summary message module_name)

  set(hpx_config_information)
  set(upper_cfg_name "HPX")
  set(upper_option_suffix "")

  string(TOUPPER ${module_name} module_name_uc)
  if(NOT "${module_name_uc}x" STREQUAL "HPXx")
    set(upper_cfg_name "HPX_${module_name_uc}")
    set(upper_option_suffix "_${module_name_uc}")
  endif()

  get_property(
    _variableNames GLOBAL PROPERTY HPX_MODULE_CONFIG_${module_name_uc}
  )
  list(SORT _variableNames)

  # Only print the module configuration if options specified
  list(LENGTH _variableNames _length)
  if(${_length} GREATER_EQUAL 1)
    hpx_info("")
    hpx_info(${message})

    get_property(
      DEFINITIONS_VARS GLOBAL
      PROPERTY HPX_CONFIG_DEFINITIONS${upper_option_suffix}
    )
    if(DEFINED DEFINITIONS_VARS)
      list(SORT DEFINITIONS_VARS)
      list(REMOVE_DUPLICATES DEFINITIONS_VARS)
    endif()

    foreach(_variableName ${_variableNames})
      if(${_variableName}Category)

        # handle only options which start with HPX_WITH_
        string(FIND ${_variableName} "${upper_cfg_name}_WITH_" __pos)

        if(${__pos} EQUAL 0)
          get_property(
            _value
            CACHE "${_variableName}"
            PROPERTY VALUE
          )
          hpx_info("    ${_variableName}=${_value}")
        endif()

        string(REPLACE "_WITH_" "_HAVE_" __variableName ${_variableName})
        list(FIND DEFINITIONS_VARS ${__variableName} __pos)
        if(NOT ${__pos} EQUAL -1)
          set(hpx_config_information
              "${hpx_config_information}"
              "\n        \"${_variableName}=${_value}\","
          )
        elseif(NOT ${_variableName}Category STREQUAL "Generic"
               AND NOT ${_variableName}Category STREQUAL "Build Targets"
        )
          get_property(
            _type
            CACHE "${_variableName}"
            PROPERTY TYPE
          )
          if(_type STREQUAL "BOOL")
            set(hpx_config_information "${hpx_config_information}"
                                       "\n        \"${_variableName}=OFF\","
            )
          endif()
        endif()
      endif()
    endforeach()
  endif()

  if(hpx_config_information)
    string(REPLACE ";" "" hpx_config_information ${hpx_config_information})
  endif()

  if("${module_name}" STREQUAL "hpx")
    set(_base_dir_local "hpx/config")
    set(_base_dir "hpx/config")
    set(_template "config_defines_strings.hpp.in")

    configure_file(
      "${HPX_SOURCE_DIR}/cmake/templates/${_template}"
      "${HPX_BINARY_DIR}/libs/core/config/include/${_base_dir_local}/config_strings.hpp"
      @ONLY
    )
    configure_file(
      "${HPX_SOURCE_DIR}/cmake/templates/${_template}"
      "${HPX_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${_base_dir}/config_strings.hpp"
      @ONLY
    )
  elseif(hpx_config_information)
    set(_base_dir_local "libs/${module_name}/src/")
    set(_template "config_defines_entries_for_modules.cpp.in")

    configure_file(
      "${HPX_SOURCE_DIR}/cmake/templates/${_template}"
      "${CMAKE_CURRENT_BINARY_DIR}/src/config_entries.cpp" @ONLY
    )
  endif()
endfunction()
