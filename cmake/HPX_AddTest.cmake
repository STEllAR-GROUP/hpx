# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_ADDTEST_LOADED TRUE)

include(HPX_Include)

hpx_include(Message
            ParseArguments)

macro(add_hpx_test name)
  # retrieve arguments
  hpx_parse_arguments(${name}
    "SOURCES;HEADERS;DEPENDENCIES;ARGS" "DONTRUN;DONTCOMPILE" ${ARGN})

  hpx_print_list("DEBUG" "add_test.${name}" "Sources for ${name}" ${name}_SOURCES)
  hpx_print_list("DEBUG" "add_test.${name}" "Headers for ${name}" ${name}_HEADERS)
  hpx_print_list("DEBUG" "add_test.${name}" "Dependencies for ${name}" ${name}_DEPENDENCIES)
  hpx_print_list("DEBUG" "add_test.${name}" "Arguments for ${name}" ${name}_ARGS)

  if(NOT ${name}_DONTCOMPILE)
    add_executable(${name}_test EXCLUDE_FROM_ALL
      ${${name}_SOURCES} ${${name}_HEADERS})
  
    set_property(TARGET ${name}_test APPEND
                 PROPERTY COMPILE_DEFINITIONS
                 "HPX_APPLICATION_NAME=${name}"
                 "HPX_APPLICATION_EXPORTS")
  
    # linker instructions
    target_link_libraries(${name}_test
      ${${name}_DEPENDENCIES} 
      ${hpx_LIBRARIES}
      ${BOOST_FOUND_LIBRARIES}
      ${pxaccel_LIBRARIES})
  
    if(NOT ${name}_DONTRUN)
      add_test(${name}
        ${CMAKE_CURRENT_BINARY_DIR}/${name}_test
        ${${name}_ARGS})
    else()
      hpx_info("add_test.${name}" "Module was not specified for component.")
    endif()
  endif()
endmacro()

