# Copyright (c) 2015 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

function(add_hpx_compile_test category name)
  set(options FAILURE_EXPECTED)
  set(one_value_args SOURCE_ROOT FOLDER)
  set(multi_value_args SOURCES)

  cmake_parse_arguments(${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  set(expected FALSE)

  if(${name}_FAILURE_EXPECTED)
    set(expected TRUE)
  endif()

  add_hpx_library(
    ${name}
    SOURCE_ROOT ${${name}_SOURCE_ROOT}
    SOURCES ${${name}_SOURCES}
    EXCLUDE_FROM_ALL
    EXCLUDE_FROM_DEFAULT_BUILD
    FOLDER ${${name}_FOLDER}
    STATIC)

  add_test(NAME "${category}.${name}"
    COMMAND ${CMAKE_COMMAND}
    --build ${CMAKE_BINARY_DIR}
      --target ${name}
      --config $<CONFIGURATION>
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  if(expected)
    set_tests_properties("${category}.${name}" PROPERTIES WILL_FAIL TRUE)
  endif()

endfunction()

function(add_hpx_unit_compile_test category name)
  add_hpx_compile_test("tests.unit.${category}" ${name} ${ARGN})
endfunction()

function(add_hpx_regression_compile_test category name)
  add_hpx_compile_test("tests.regressions.${category}" ${name} ${ARGN})
endfunction()

function(add_hpx_headers_compile_test category name)
  add_hpx_compile_test("tests.headers.${category}" ${name} ${ARGN})
endfunction()

function(add_hpx_lib_header_tests lib)
  file(GLOB_RECURSE headers ${DO_CONFIGURE_DEPENDS} "${PROJECT_SOURCE_DIR}/include/hpx/*hpp")
  set(all_headers)
  add_custom_target(tests.headers.${lib})
  add_dependencies(tests.headers tests.headers.${lib})
  foreach(header ${headers})

    # skip all headers in directories containing 'detail'
    set(detail_pos -1)
    string(FIND "${header}" "detail" detail_pos)

    if(${detail_pos} EQUAL -1)
      # extract relative path of header
      string(REGEX REPLACE "${PROJECT_SOURCE_DIR}/include/hpx/" "" relpath "${header}")

      # .hpp --> .cpp
      string(REGEX REPLACE ".hpp" ".cpp" full_test_file "${relpath}")
      # remove extension, '/' --> '_'
      string(REGEX REPLACE ".hpp" "_hpp" test_file "${relpath}")
      string(REGEX REPLACE "/" "_" test_name "${test_file}")

      # generate the test
      file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${full_test_file}
        "#include <hpx/${relpath}>\n"
        "#ifndef HPX_MAIN_DEFINED\n"
        "int main(int argc, char** argv) { return 0; }\n"
        "#endif\n")

      set(all_headers ${all_headers} "#include <hpx/${relpath}>\n")

      add_library(tests.headers.${lib}.${test_name} ${CMAKE_CURRENT_BINARY_DIR}/${full_test_file})
      target_link_libraries(tests.headers.${lib}.${test_name} hpx_${lib})
      add_dependencies(tests.headers.${lib} tests.headers.${lib}.${test_name})
    endif()
  endforeach()

  set(test_name "all_headers")
  set(all_headers_test_name "${CMAKE_CURRENT_BINARY_DIR}/${test_name}.cpp")
  file(WRITE ${all_headers_test_name}
    ${all_headers}
    "#ifndef HPX_MAIN_DEFINED\n"
    "int main(int argc, char** argv) { return 0; }\n"
    "#endif\n")

  add_library(tests.headers.${lib}.${test_name} "${CMAKE_CURRENT_BINARY_DIR}/${test_name}.cpp")
  target_link_libraries(tests.headers.${lib}.${test_name} hpx_${lib})
  add_dependencies(tests.headers.${lib} tests.headers.${lib}.${test_name})
endfunction()
