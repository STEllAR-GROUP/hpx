# Copyright (c) 2025 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(HPX_Message)

# List of known standard library headers
set(STANDARD_LIBRARY_HEADERS
    "<algorithm>"
    "<array>"
    "<atomic>"
    "<bit>"
    "<bitset>"
    "<cstddef>"
    "<cstdint>"
    "<chrono>"
    "<condition_variable>"
    "<deque>"
    "<exception>"
    "<filesystem>"
    "<forward_list>"
    "<functional>"
    "<future>"
    "<iomanip>"
    "<ios>"
    "<iostream>"
    "<istream>"
    "<iterator>"
    "<limits>"
    "<list>"
    "<map>"
    "<memory>"
    "<mutex>"
    "<numeric>"
    "<ostream>"
    "<queue>"
    "<set>"
    "<shared_mutex>"
    "<stack>"
    "<string>"
    "<string_view>"
    "<system_error>"
    "<thread>"
    "<tuple>"
    "<type_traits>"
    "<unordered_map>"
    "<unordered_set>"
    "<utility>"
    "<valarray>"
    "<variant>"
    "<vector>"
)

# Function to extract #includes from a file recursively
function(hpx_extract_includes_from_file module)

  # retrieve arguments
  set(options)
  set(one_value_args SOURCE FOUND_HEADERS)
  set(multi_value_args INCLUDE_DIRS)
  cmake_parse_arguments(
    ${module} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN}
  )

  # look at every file only once
  set(this_file ${${module}_SOURCE})
  get_property(current_checked_list GLOBAL PROPERTY checked_list)
  if(${this_file} IN_LIST current_checked_list)
    return()
  endif()
  list(APPEND current_checked_list "${this_file}")
  set_property(GLOBAL PROPERTY checked_list "${current_checked_list}")

  file(READ ${this_file} file_content)

  # Regex to match #include directives for standard headers
  string(REGEX MATCHALL "#include <[^>]+>" includes ${file_content})

  set(found_includes)
  foreach(include ${includes})
    string(REGEX REPLACE "#include (<[^>]+>)" "\\1" filename ${include})

    if(NOT filename MATCHES "\\.|/")
      # Check if the include is a standard library header
      if(${filename} IN_LIST STANDARD_LIBRARY_HEADERS)
        list(APPEND found_includes ${filename})
      endif()
    endif()
  endforeach()

  # Regex to match #include directives for hpx/ files
  string(REGEX MATCHALL "#include <hpx/[^>]+>" hpx_includes ${file_content})

  foreach(hpx_include ${hpx_includes})
    # Extract the file name from the hpx angle include
    string(REGEX REPLACE "#include <(hpx/[^>]+)>" "\\1" hpx_file ${hpx_include})

    # Check if the file exists in any of the base directories
    set(file_found FALSE)
    foreach(base_dir ${${module}_INCLUDE_DIRS})

      set(full_filename "${base_dir}/${hpx_file}")
      if(EXISTS "${full_filename}")
        set(file_found TRUE)

        # Recursively extract includes from the hpx file
        set(found_recursive_includes "")
        hpx_extract_includes_from_file(
          ${module}
          SOURCE "${full_filename}"
          INCLUDE_DIRS ${${module}_INCLUDE_DIRS}
          FOUND_HEADERS found_recursive_includes
        )
        list(APPEND found_includes ${found_recursive_includes})

        break()
      endif()
    endforeach()

    if(NOT file_found)
      hpx_warn("Included file does not exist: ${hpx_file}")
    endif()
  endforeach()

  set(${${module}_FOUND_HEADERS}
      ${found_includes}
      PARENT_SCOPE
  )
endfunction()

function(hpx_collect_std_headers module)
  # retrieve arguments
  set(options)
  set(one_value_args SOURCE_ROOT GENERATED_ROOT FOUND_HEADERS)
  set(multi_value_args SOURCES)
  cmake_parse_arguments(
    ${module} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN}
  )

  if(NOT ${module}_SOURCE_ROOT)
    hpx_error("SOURCE_ROOT argument not specified")
  endif()
  if(NOT ${module}_FOUND_HEADERS)
    hpx_error("FOUND_HEADERS argument not specified")
  endif()
  if(NOT ${module}_SOURCES)
    hpx_error("SOURCES argument not specified")
  endif()

  set_property(GLOBAL PROPERTY checked_list)

  # Generate a list of base directories that end with '/include'
  file(
    GLOB_RECURSE header_dirs
    LIST_DIRECTORIES true
    "${${module}_SOURCE_ROOT}/**/include/"
  )

  if(${module}_GENERATED_ROOT)
    file(
      GLOB_RECURSE generated_header_dirs
      LIST_DIRECTORIES true
      "${${module}_GENERATED_ROOT}/**/include/"
    )
    list(APPEND header_dirs ${generated_header_dirs})
  endif()

  # Make sure only to consider directories ending with 'include/'
  set(include_dirs)
  foreach(item ${header_dirs})
    # Check if the item matches the regex and add it to the filtered list
    string(REGEX MATCH "include$" match_result "${item}")
    if(match_result)
      list(APPEND include_dirs "${item}")
    endif()
  endforeach()

  # Loop through each source file and extract includes
  set(found_includes)
  foreach(source_file ${${module}_SOURCES})
    set(found_recursive_includes)
    hpx_extract_includes_from_file(
      ${module}
      SOURCE "include/${source_file}"
      INCLUDE_DIRS ${include_dirs}
      FOUND_HEADERS found_recursive_includes
    )
    list(APPEND found_includes ${found_recursive_includes})
  endforeach()

  set(${${module}_FOUND_HEADERS}
      ${found_includes}
      PARENT_SCOPE
  )
endfunction()
