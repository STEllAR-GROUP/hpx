# Copyright (c) 2014 Thomas Heller
# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_ADDCONFIGTEST_LOADED TRUE)

macro(add_hpx_config_test variable)
  set(options FILE)
  set(one_value_args SOURCE ROOT)
  set(multi_value_args INCLUDE_DIRECTORIES LINK_DIRECTORIES COMPILE_DEFINITIONS LIBRARIES ARGS DEFINITIONS)
  cmake_parse_arguments(${variable} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(NOT DEFINED ${variable})
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/config_tests")

    string(TOUPPER "${variable}" variable_lc)
    if(${variable}_FILE)
      if(${variable}_ROOT)
        set(test_source "${${variable}_ROOT}/share/hpx-${HPX_VERSION}/${${variable}_SOURCE}")
      else()
        set(test_source "${hpx_SOURCE_DIR}/${${variable}_SOURCE}")
      endif()
    else()
      set(test_source
          "${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/config_tests/${variable_lc}.cpp")
      file(WRITE "${test_source}"
           "${${variable}_SOURCE}\n")
    endif()
    set(test_binary ${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/config_tests/${variable_lc})

    get_directory_property(CONFIG_TEST_INCLUDE_DIRS INCLUDE_DIRECTORIES)
    get_directory_property(CONFIG_TEST_LINK_DIRS LINK_DIRECTORIES)
    set(COMPILE_DEFINITIONS_TMP)
    set(CONFIG_TEST_COMPILE_DEFINITIONS)
    get_directory_property(COMPILE_DEFINITIONS_TMP COMPILE_DEFINITIONS)
    foreach(def ${COMPILE_DEFINITIONS_TMP})
      set(CONFIG_TEST_COMPILE_DEFINITIONS "${CONFIG_TEST_COMPILE_DEFINITIONS} -D${def}")
    endforeach()

    set(CONFIG_TEST_INCLUDE_DIRS ${CONFIG_TEST_INCLUDE_DIRS} ${${variable}_INCLUDE_DIRS})
    set(CONFIG_TEST_LINK_DIRS ${CONFIG_TEST_LINK_DIRS} ${${variable}_LINK_DIRS})

    set(CONFIG_TEST_COMPILE_DEFINITIONS ${CONFIG_TEST_COMPILE_DEFINITIONS} ${${variable}_COMPILE_DEFINITIONS})
    set(CONFIG_TEST_LINK_LIBRARIES ${HPX_LIBRARIES} ${${variable}_LIBRARIES})

    try_compile(${variable}_RESULT
      ${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/config_tests
      ${test_source}
      CMAKE_FLAGS
        "-DINCLUDE_DIRECTORIES=${CONFIG_TEST_INCLUDE_DIRS}"
        "-DLINK_DIRECTORIES=${CONFIG_TEST_LINK_DIRS}"
        "-DLINK_LIBRARIES=${CONFIG_TEST_LINK_LIBRARIES}"
        "-DCOMPILE_DEFINITIONS=${CONFIG_TEST_COMPILE_DEFINITIONS}"
      OUTPUT_VARIABLE ${variable}_OUTPUT
      COPY_FILE ${test_binary})

    string(TOUPPER "${variable}" variable_uc)
    set(_msg "Performing Test ${variable_uc}")

    if(${variable}_RESULT)
      set(_run_msg "Success")
      #if(NOT CMAKE_CROSSCOMPILING)
      #  execute_process(
      #    COMMAND "${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/config_tests/${variable_lc}"
      #      ${${variable}_ARGS}
      #      RESULT_VARIABLE test_result OUTPUT_QUIET ERROR_QUIET)
      #  if(NOT test_result STREQUAL "0")
      #    set(${variable}_RESULT OFF)
      #    set(_run_msg "Failed executing.")
      #  endif()
      #endif()

      set(_msg "${_msg} - ${_run_msg}")
    else()
      set(_msg "${_msg} - Failed")
      if(NOT MSVC)
        set(_msg "${_msg} \n ${${variable}_OUTPUT}")
      endif()
    endif()

    set(${variable} ${${variable}_RESULT} CACHE BOOL INTERNAL)
    hpx_info(${_msg})
  else()
    set(${variable}_RESULT ${${variable}})
  endif()

  if(${variable}_RESULT)
    foreach(definition ${${variable}_DEFINITIONS})
      hpx_add_config_define(${definition})
    endforeach()
  endif()
endmacro()

###############################################################################
macro(hpx_cpuid target variable)
  add_hpx_config_test(${variable}
    SOURCE cmake/tests/cpuid.cpp
    FLAGS "${boost_include_dir}" "${include_dir}"
    FILE ARGS "${target}" ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_rvalue_references)
  add_hpx_config_test(HPX_WITH_CXX11_RVALUE_REFERENCES
    SOURCE cmake/tests/cxx11_rvalue_references.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_variadic_templates)
  add_hpx_config_test(HPX_WITH_CXX11_VARIADIC_TEMPLATES
    SOURCE cmake/tests/cxx11_variadic_templates.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_lambdas)
  add_hpx_config_test(HPX_WITH_CXX11_LAMBDAS
    SOURCE cmake/tests/cxx11_lambdas.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_auto)
  add_hpx_config_test(HPX_WITH_CXX11_AUTO
    SOURCE cmake/tests/cxx11_auto.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_decltype)
  add_hpx_config_test(HPX_WITH_CXX11_DECLTYPE
    SOURCE cmake/tests/cxx11_decltype.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_alias_templates)
  add_hpx_config_test(HPX_WITH_CXX11_ALIAS_TEMPLATES
    SOURCE cmake/tests/cxx11_alias_templates.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_defaulted_deleted_functions)
  add_hpx_config_test(HPX_WITH_CXX11_DEFAULTED_DELETED_FUNCTIONS
    SOURCE cmake/tests/cxx11_defaulted_deleted_functions.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_std_unique_ptr)
  add_hpx_config_test(HPX_WITH_CXX11_UNIQUE_PTR
    SOURCE cmake/tests/cxx11_std_unique_ptr.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_std_tuple)
  add_hpx_config_test(HPX_WITH_CXX11_STD_TUPLE
    SOURCE cmake/tests/cxx11_std_tuple.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_std_function)
  add_hpx_config_test(HPX_WITH_CXX11_STD_FUNCTION
    SOURCE cmake/tests/cxx11_std_function.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_std_initializer_list)
  add_hpx_config_test(HPX_WITH_CXX11_STD_INITIALIZER_LIST
    SOURCE cmake/tests/cxx11_std_initializer_list.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_std_chrono)
  add_hpx_config_test(HPX_WITH_CXX11_CHRONO
    SOURCE cmake/tests/cxx11_std_chrono.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_thread_safe_hdf5)
  add_hpx_config_test(WITH_HDF5_THREAD_SAFE
    SOURCE cmake/tests/hdf5_thread_safe.cpp
    FILE ${ARGN})
endmacro()

