# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_ADDCONFIGTEST_LOADED TRUE)

include(HPX_Include)

hpx_include(Message
            ParseArguments)

macro(add_hpx_config_test name var)
  hpx_parse_arguments(${name} "SOURCE;FLAGS" "ESSENTIAL" ${ARGN})

  file(WRITE "${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/CMakeTmp/src.cpp"
       "${${name}_SOURCE}\n")

  try_compile(${var}
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/CMakeTmp/src.cpp
    CMAKE_FLAGS -DCOMPILE_DEFINITIONS:STRING=${${name}_FLAGS}
    OUTPUT_VARIABLE output)

  if(${var})
    set(${var} TRUE CACHE INTERNAL "Test ${name} result.")
    hpx_info("config_test.${name}" "Test passed.")
    file(APPEND ${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/CMakeOutput.log
      "Test ${name} passed with the following output:\n"
      "${output}\n"
      "Source code was:\n${${name}_SOURCE}\n")
  else()
    set(${var} FALSE CACHE INTERNAL "Test ${name} result.")
    if(${name}_ESSENTIAL)
      hpx_fail("config_test.${name}" "Test failed (check ${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/CMakeError.log).")
    else()
      hpx_warn("config_test.${name}" "Test failed (check ${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/CMakeError.log).")
    endif()
    file(APPEND ${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/CMakeError.log
      "Test ${name} failed with the following output:\n"
      "${output}\n"
      "Source code was:\n${${name}_SOURCE}\n")
  endif()
endmacro()

###############################################################################
macro(hpx_check_pthreads_affinity variable)
  add_hpx_config_test(
   "pthreads_affinity"
   ${variable}
   SOURCE
   "#include <pthread.h>
    
    int f()
    {
        pthread_t th;
        size_t cpusetsize;
        cpu_set_t* cpuset;
        pthread_setaffinity_np(th, cpusetsize, cpuset);
        pthread_getaffinity_np(th, cpusetsize, cpuset);
    }
    
    int main()
    {
        return 0;
    }"
    FLAGS -pthread)
endmacro()

