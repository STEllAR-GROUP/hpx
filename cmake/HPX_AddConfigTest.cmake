# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_ADDCONFIGTEST_LOADED TRUE)

include(HPX_Include)

hpx_include(Message
            Compile
            ParseArguments)

macro(add_hpx_config_test name variable)
  hpx_parse_arguments(${name} "SOURCE;FLAGS;DEFINITIONS;LANGUAGE;DEFAULT;ARGS"
                              "FILE" ${ARGN})

  # FIXME: Sadly, CMake doesn't support non-boolean options with the option
  # command yet.
  #option(${variable}
  #  "Enable (ON), auto-detect (DETECT) or disable (OFF) ${name} (Default: ${${name}_DEFAULT}"
  #  ${${name}_DEFAULT})

  if("${variable}" STREQUAL "ON")
    hpx_info("config_test.${name}" "Manually enabled.")
    set(${variable} TRUE CACHE INTERNAL "${name} state.")
    foreach(definition ${${name}_DEFINITIONS})
      add_definitions(-D${definition})
    endforeach()
  elseif("${variable}" STREQUAL "OFF")
    hpx_info("config_test.${name}" "Manually disabled.")
    set(${variable} FALSE CACHE INTERNAL "${name} state.")
  else()
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
                ${${name}_ARGS}
        RESULT_VARIABLE test_result OUTPUT_QUIET ERROR_QUIET) 
  
      if("${test_result}" STREQUAL "0")
        set(${variable} TRUE CACHE INTERNAL "${name} state.")
        hpx_info("config_test.${name}" "Test passed.")
        foreach(definition ${${name}_DEFINITIONS})
          add_definitions(-D${definition})
        endforeach()
      else()
        set(${variable} FALSE CACHE INTERNAL "${name} state.")
        hpx_warn("config_test.${name}" "Test failed, returned ${test_result}.") 
      endif()
    else()
      set(${variable} FALSE CACHE INTERNAL "${name} state.")
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

macro(hpx_check_for_gnu_aligned_16 variable)
  add_hpx_config_test("gnu_aligned_16" ${variable} LANGUAGE CXX 
    SOURCE ${hpx_SOURCE_DIR}/cmake/tests/gnu_aligned_16.cpp
    FLAGS -I ${BOOST_INCLUDE_DIR} -I ${hpx_SOURCE_DIR} FILE ${ARGN})
endmacro()

macro(hpx_check_for_gnu_mcx16 variable)
  add_hpx_config_test("gnu_mcx16" ${variable} LANGUAGE CXX 
    SOURCE ${hpx_SOURCE_DIR}/cmake/tests/flag.cpp
    FLAGS -mcx16 FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_pthread_affinity_np variable)
  add_hpx_config_test("pthread_affinity_np" ${variable} LANGUAGE CXX 
    SOURCE ${hpx_SOURCE_DIR}/cmake/tests/pthread_affinity_np.cpp
    FLAGS -pthread -I ${BOOST_INCLUDE_DIR} -I ${hpx_SOURCE_DIR} FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_compiler_auto_tune variable)
  # TODO: add support for MSVC-esque compilers
  add_hpx_config_test("compiler_auto_tune" ${variable} LANGUAGE CXX 
    SOURCE ${hpx_SOURCE_DIR}/cmake/tests/flag.cpp
    FLAGS -march=native FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_cpuid target variable)
  add_hpx_config_test("${target}" ${variable} LANGUAGE CXX 
    SOURCE ${hpx_SOURCE_DIR}/cmake/tests/cpuid.cpp
    FLAGS -I ${BOOST_INCLUDE_DIR} -I ${hpx_SOURCE_DIR} FILE ${ARGN}
    ARGS "${target}")
endmacro()

