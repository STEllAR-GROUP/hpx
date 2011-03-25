# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_ADDCONFIGTEST_LOADED TRUE)

include(HPX_Include)

hpx_include(Message
            Compile
            ParseArguments)

macro(add_hpx_config_test name var)
  hpx_parse_arguments(${name} "SOURCE;FLAGS;DEFINITIONS;LANGUAGE"
                              "ESSENTIAL;FILE" ${ARGN})

  file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/config_tests)

  set(test_source "")

  if(${name}_FILE)
    set(test_source "${${name}_SOURCE}")
  else()
    set(test_source
        "${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/config_tests/src.cpp")
    file(WRITE "${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/config_tests/src.cpp"
         "${${name}_SOURCE}\n")
  endif()

  hpx_compile(${name} SOURCE ${test_source} LANGUAGE ${${name}_LANGUAGE}
    OUTPUT ${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/config_tests/${name} 
    FLAGS ${${name}_FLAGS})

  if("${${name}_RESULT}" STREQUAL "0")
    set(test_result 0)
  
    execute_process(
      COMMAND "${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/config_tests/${name}"
      RESULT_VARIABLE test_result OUTPUT_QUIET ERROR_QUIET) 

    if("${test_result}" STREQUAL "0")
      set(${var} TRUE CACHE INTERNAL "Test ${name} result.")
      hpx_info("config_test.${name}" "Test passed.")
      foreach(definition ${${name}_DEFINITIONS})
        add_definitions(-D${definition})
      endforeach()
    else()
      set(${var} FALSE CACHE INTERNAL "Test ${name} result.")
      if(${name}_ESSENTIAL)
        hpx_fail("config_test.${name}" "Test failed, returned ${test_result}.") 
      else()
        hpx_warn("config_test.${name}" "Test failed, returned ${test_result}.") 
      endif()
    endif()
  else()
    set(${var} FALSE CACHE INTERNAL "Test ${name} result.")
    if(${name}_ESSENTIAL)
      hpx_fail("config_test.${name}" "Test failed to compile.") 
    else()
      hpx_warn("config_test.${name}" "Test failed to compile.") 
    endif()
  endif()
endmacro()

###############################################################################
macro(hpx_check_for_gnu_128bit_integers variable)
  add_hpx_config_test("gnu_int128" ${variable} LANGUAGE CXX
    SOURCE ${hpx_SOURCE_DIR}/cmake/tests/gnu_128bit_integers.cpp
    FLAGS -I ${BOOST_INCLUDE_DIR} -I ${hpx_SOURCE_DIR} FILE ${ARGN})
endmacro()

macro(hpx_check_for_gnu_mcx16 variable)
  add_hpx_config_test("gnu_mcx16" ${variable} LANGUAGE CXX
    SOURCE ${hpx_SOURCE_DIR}/cmake/tests/flag.cpp
    FLAGS -mcx16 FILE ${ARGN})
endmacro()

macro(hpx_check_for_gnu_march variable)
  add_hpx_config_test("gnu_march" ${variable} LANGUAGE CXX
    SOURCE ${hpx_SOURCE_DIR}/cmake/tests/flag.cpp
    FLAGS -march=native FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_pthread_affinity_np variable)
  add_hpx_config_test("pthread_affinity_np" ${variable} LANGUAGE CXX
    SOURCE ${hpx_SOURCE_DIR}/cmake/tests/pthread_affinity_np.cpp
    FLAGS -pthread -I ${BOOST_INCLUDE_DIR} -I ${hpx_SOURCE_DIR} FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_rdtscp variable)
  add_hpx_config_test("rdtscp" ${variable} LANGUAGE CXX
    SOURCE ${hpx_SOURCE_DIR}/cmake/tests/rdtscp.cpp
    FLAGS -I ${BOOST_INCLUDE_DIR} -I ${hpx_SOURCE_DIR} FILE ${ARGN})
endmacro()

