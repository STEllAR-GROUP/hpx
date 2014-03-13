# Copyright (c) 2014 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_STATIC_COMPONENTLIST_LOADED TRUE)

include(HPX_Include)

hpx_include(Message)

if(NOT HPX_STATIC_LINKING)
  macro(hpx_static_componentlist)
  endmacro()
else()
  macro(hpx_static_componentlist)
    # build the executable
    add_executable(create_static_module_data_exe
      create_static_module_data.cpp)
    set_target_properties(create_static_module_data_exe
      PROPERTIES
       RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
       FOLDER "Tools/create_static_module_data_exe")
    add_hpx_pseudo_dependencies(tools create_static_module_data_exe)

    # create custom build step for generating the file static_component_data.hpp
    set(component_list_files
      ${hpx_SOURCE_DIR}/src/components/static_components.list)
    set(static_component_data_dependencies
      ${hpx_SOURCE_DIR}/cmake/templates/static_component_data.hpp.in)

    if(MSVC)
      set(create_static_module_data_dir
        "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}")
    else()
      set(create_static_module_data_dir
        "${CMAKE_CURRENT_BINARY_DIR}")
    endif()

    hpx_debug("hpx_static_componentlist"
        "create_static_module_data_dir: ${create_static_module_data_dir}")

    add_custom_command(
      OUTPUT ${hpx_SOURCE_DIR}/hpx/components/static_component_data.hpp
      COMMAND
          "${create_static_module_data_dir}/create_static_module_data_exe"
          "${hpx_SOURCE_DIR}/cmake/templates/static_component_data.hpp.in"
          "${hpx_SOURCE_DIR}/hpx/components/static_component_data.hpp"
          "${component_list_files}"
      COMMENT "Generating static component list."
      DEPENDS create_static_module_data_exe ${static_component_data_dependencies})

    set(static_component_data_files
      ${hpx_SOURCE_DIR}/cmake/templates/static_component_data.hpp.in
      ${hpx_SOURCE_DIR}/src/components/static_components.list
      ${hpx_SOURCE_DIR}/hpx/components/static_component_data.hpp)

    add_custom_target(static_component_data_hpp
      DEPENDS ${hpx_SOURCE_DIR}/hpx/components/static_component_data.hpp
      SOURCES ${static_component_data_files})

    # add files to created project file
    if(MSVC)
      hpx_print_list("DEBUG" "hpx_static_componentlist"
        "static_component_data_files" static_component_data_files)

      source_group("Files" FILES ${static_component_data_files})
      set_source_files_properties(${static_component_data_files}
        HEADER_FILE_ONLY TRUE)
    endif()
  endmacro()
endif()
