# Copyright (c) 2022 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Generate HPX Dependency Report

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

separate_arguments(HPX_CORE_ENABLED_MODULES)
separate_arguments(HPX_FULL_ENABLED_MODULES)

# common options
set(hpxdep_common_options --hpx-root "${HPX_SOURCE_DIR}" --hpx-build-root
                          "${HPX_BINARY_DIR}"
)

set(HPXDEP_OUTPUT_DIR
    "${HPX_BINARY_DIR}/share/hpx/docs/report/${HPX_DEPREPORT_VERSION}"
)

# create index.html
configure_file(
  "${HPX_SOURCE_DIR}/cmake/templates/hpxdep_index.html.in"
  "${HPX_BINARY_DIR}/share/hpx/docs/report/index.html" @ONLY
)

# create module-list
set(hpxdep_module_list_output "module-list.txt")
message("Generating ${hpxdep_module_list_output}")

set(module_list)
foreach(module ${HPX_CORE_ENABLED_MODULES})
  set(module_list ${module_list} "core/${module}\n")
endforeach()
foreach(module ${HPX_FULL_ENABLED_MODULES})
  set(module_list ${module_list} "full/${module}\n")
endforeach()
file(MAKE_DIRECTORY "${HPXDEP_OUTPUT_DIR}")
file(WRITE "${HPXDEP_OUTPUT_DIR}/${hpxdep_module_list_output}" ${module_list})

# module-overview, module-levels, and module-weights reports
macro(generate_module_overview_report option)
  string(SUBSTRING ${option} 0 1 first_letter)
  string(TOUPPER ${first_letter} first_letter)
  string(REGEX REPLACE "^.(.*)" "${first_letter}\\1" option_cap "${option}")

  set(hpxdep_module_${option}_output "module-${option}.html")
  message("Generating ${hpxdep_module_${option}_output}")

  # cmake-format: off
  set(hpxdep_module_${option}_build_command
      ${HPXDEP_OUTPUT_NAME}
          ${hpxdep_common_options}
          --module-list "${HPXDEP_OUTPUT_DIR}/${hpxdep_module_list_output}"
          --html-title "\"HPX Module ${option_cap}\""
          --html
          --module-${option}
  )
  # cmake-format: on

  execute_process(
    COMMAND ${hpxdep_module_${option}_build_command}
    WORKING_DIRECTORY "${HPXDEP_OUTPUT_DIR}"
    OUTPUT_FILE "${HPXDEP_OUTPUT_DIR}/${hpxdep_module_${option}_output}"
    RESULT_VARIABLE hpxdep_result
    ERROR_VARIABLE hpxdep_error
  )
  if(NOT "${hpxdep_result}" EQUAL "0")
    message(
      FATAL_ERROR
        "Generating ${hpxdep_module_${option}_output} failed: ${hpxdep_error}."
    )
  endif()
endmacro()

generate_module_overview_report("overview")
generate_module_overview_report("levels")
generate_module_overview_report("weights")

# module-specific reports
macro(generate_module_report libname)
  file(MAKE_DIRECTORY "${HPXDEP_OUTPUT_DIR}/${libname}")
  foreach(module ${ARGN})
    set(module_name "${libname}/${module}")
    set(hpxdep_module_output "${module_name}.html")
    message("Generating ${hpxdep_module_output}")

    # cmake-format: off
    set(hpxdep_module_build_command
        ${HPXDEP_OUTPUT_NAME}
            ${hpxdep_common_options}
            --module-list "${HPXDEP_OUTPUT_DIR}/${hpxdep_module_list_output}"
            --html-title "\"HPX Dependency Report for ${module_name}\""
            --html
            --primary ${module_name}
            --secondary ${module_name}
            --reverse ${module_name}
    )
    # cmake-format: on

    execute_process(
      COMMAND ${hpxdep_module_build_command}
      WORKING_DIRECTORY "${HPXDEP_OUTPUT_DIR}"
      OUTPUT_FILE "${HPXDEP_OUTPUT_DIR}/${hpxdep_module_output}"
      RESULT_VARIABLE hpxdep_result
      ERROR_VARIABLE hpxdep_error
    )
    if(NOT "${hpxdep_result}" EQUAL "0")
      message(
        FATAL_ERROR
          "Generating ${hpxdep_module_output} failed: ${hpxdep_error}."
      )
    endif()
  endforeach()
endmacro()

generate_module_report("core" ${HPX_CORE_ENABLED_MODULES})
generate_module_report("full" ${HPX_FULL_ENABLED_MODULES})
