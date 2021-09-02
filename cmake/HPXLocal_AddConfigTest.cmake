# Copyright (c) 2011 Bryce Lelbach
# Copyright (c) 2014 Thomas Heller
# Copyright (c) 2017 Denis Blank
# Copyright (c) 2017 Google
# Copyright (c) 2017 Taeguk Kwon
# Copyright (c) 2020 Giannis Gonidelis
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPXLocal_ADDCONFIGTEST_LOADED TRUE)

include(CheckLibraryExists)

function(hpx_local_add_config_test variable)
  set(options FILE EXECUTE CUDA)
  set(one_value_args SOURCE ROOT CMAKECXXFEATURE)
  set(multi_value_args
      INCLUDE_DIRECTORIES
      LINK_DIRECTORIES
      COMPILE_DEFINITIONS
      LIBRARIES
      ARGS
      DEFINITIONS
      REQUIRED
  )
  cmake_parse_arguments(
    ${variable} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN}
  )

  set(_run_msg)
  # Check CMake feature tests if the user didn't override the value of this
  # variable:
  if(NOT DEFINED ${variable} AND NOT ${variable}_CUDA)
    if(${variable}_CMAKECXXFEATURE)
      # We don't have to run our own feature test if there is a corresponding
      # cmake feature test and cmake reports the feature is supported on this
      # platform.
      list(FIND CMAKE_CXX_COMPILE_FEATURES ${${variable}_CMAKECXXFEATURE} __pos)
      if(NOT ${__pos} EQUAL -1)
        set(${variable}
            TRUE
            CACHE INTERNAL ""
        )
        set(_run_msg "Success (cmake feature test)")
      endif()
    endif()
  endif()

  if(NOT DEFINED ${variable})
    file(MAKE_DIRECTORY
         "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/config_tests"
    )

    string(TOLOWER "${variable}" variable_lc)
    if(${variable}_FILE)
      if(${variable}_ROOT)
        set(test_source "${${variable}_ROOT}/share/hpx/${${variable}_SOURCE}")
      else()
        set(test_source "${PROJECT_SOURCE_DIR}/${${variable}_SOURCE}")
      endif()
    else()
      if(${variable}_CUDA)
        set(test_source
            "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/config_tests/${variable_lc}.cu"
        )
      else()
        set(test_source
            "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/config_tests/${variable_lc}.cpp"
        )
      endif()
      file(WRITE "${test_source}" "${${variable}_SOURCE}\n")
    endif()
    set(test_binary
        ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/config_tests/${variable_lc}
    )

    get_directory_property(CONFIG_TEST_INCLUDE_DIRS INCLUDE_DIRECTORIES)
    get_directory_property(CONFIG_TEST_LINK_DIRS LINK_DIRECTORIES)
    set(COMPILE_DEFINITIONS_TMP)
    set(CONFIG_TEST_COMPILE_DEFINITIONS)
    get_directory_property(COMPILE_DEFINITIONS_TMP COMPILE_DEFINITIONS)
    foreach(def IN LISTS COMPILE_DEFINITIONS_TMP
                         ${variable}_COMPILE_DEFINITIONS
    )
      set(CONFIG_TEST_COMPILE_DEFINITIONS
          "${CONFIG_TEST_COMPILE_DEFINITIONS} -D${def}"
      )
    endforeach()
    get_property(
      HPXLocal_TARGET_COMPILE_OPTIONS_PUBLIC_VAR GLOBAL
      PROPERTY HPXLocal_TARGET_COMPILE_OPTIONS_PUBLIC
    )
    get_property(
      HPXLocal_TARGET_COMPILE_OPTIONS_PRIVATE_VAR GLOBAL
      PROPERTY HPXLocal_TARGET_COMPILE_OPTIONS_PRIVATE
    )
    set(HPXLocal_TARGET_COMPILE_OPTIONS_VAR
        ${HPXLocal_TARGET_COMPILE_OPTIONS_PUBLIC_VAR}
        ${HPXLocal_TARGET_COMPILE_OPTIONS_PRIVATE_VAR}
    )
    foreach(_flag ${HPXLocal_TARGET_COMPILE_OPTIONS_VAR})
      if(NOT "${_flag}" MATCHES "^\\$.*")
        set(CONFIG_TEST_COMPILE_DEFINITIONS
            "${CONFIG_TEST_COMPILE_DEFINITIONS} ${_flag}"
        )
      endif()
    endforeach()

    set(CONFIG_TEST_INCLUDE_DIRS ${CONFIG_TEST_INCLUDE_DIRS}
                                 ${${variable}_INCLUDE_DIRECTORIES}
    )
    set(CONFIG_TEST_LINK_DIRS ${CONFIG_TEST_LINK_DIRS}
                              ${${variable}_LINK_DIRECTORIES}
    )

    set(CONFIG_TEST_LINK_LIBRARIES ${${variable}_LIBRARIES})

    set(additional_cmake_flags)
    if(MSVC)
      set(additional_cmake_flags "-WX")
    else()
      set(additional_cmake_flags "-Werror")
    endif()

    if(${variable}_EXECUTE)
      if(NOT CMAKE_CROSSCOMPILING)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${additional_cmake_flags}")
        # cmake-format: off
        try_run(
          ${variable}_RUN_RESULT ${variable}_COMPILE_RESULT
          ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/config_tests
          ${test_source}
          COMPILE_DEFINITIONS ${CONFIG_TEST_COMPILE_DEFINITIONS}
          CMAKE_FLAGS
            "-DINCLUDE_DIRECTORIES=${CONFIG_TEST_INCLUDE_DIRS}"
            "-DLINK_DIRECTORIES=${CONFIG_TEST_LINK_DIRS}"
            "-DLINK_LIBRARIES=${CONFIG_TEST_LINK_LIBRARIES}"
          CXX_STANDARD ${HPXLocal_CXX_STANDARD}
          CXX_STANDARD_REQUIRED ON
          CXX_EXTENSIONS FALSE
          RUN_OUTPUT_VARIABLE ${variable}_OUTPUT
          ARGS ${${variable}_ARGS}
        )
        # cmake-format: on
        if(${variable}_COMPILE_RESULT AND NOT ${variable}_RUN_RESULT)
          set(${variable}_RESULT TRUE)
        else()
          set(${variable}_RESULT FALSE)
        endif()
      else()
        set(${variable}_RESULT FALSE)
      endif()
    else()
      if(HPXLocal_WITH_CUDA)
        set(cuda_parameters CUDA_STANDARD ${CMAKE_CUDA_STANDARD})
      endif()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${additional_cmake_flags}")
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${additional_cmake_flags}")
      # cmake-format: off
      try_compile(
        ${variable}_RESULT
        ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/config_tests
        ${test_source}
        COMPILE_DEFINITIONS ${CONFIG_TEST_COMPILE_DEFINITIONS}
        CMAKE_FLAGS
          "-DINCLUDE_DIRECTORIES=${CONFIG_TEST_INCLUDE_DIRS}"
          "-DLINK_DIRECTORIES=${CONFIG_TEST_LINK_DIRS}"
          "-DLINK_LIBRARIES=${CONFIG_TEST_LINK_LIBRARIES}"
        OUTPUT_VARIABLE ${variable}_OUTPUT
        CXX_STANDARD ${HPXLocal_CXX_STANDARD}
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS FALSE
        ${cuda_parameters}
        COPY_FILE ${test_binary}
      )
      # cmake-format: on
      hpx_local_debug("Compile test: ${variable}")
      hpx_local_debug("Compilation output: ${${variable}_OUTPUT}")
    endif()

    set(_run_msg "Success")
  else()
    set(${variable}_RESULT ${${variable}})
    if(NOT _run_msg)
      set(_run_msg "pre-set to ${${variable}}")
    endif()
  endif()

  set(_msg "Performing Test ${variable}")

  if(${variable}_RESULT)
    set(_msg "${_msg} - ${_run_msg}")
  else()
    set(_msg "${_msg} - Failed")
  endif()

  set(${variable}
      ${${variable}_RESULT}
      CACHE INTERNAL ""
  )
  hpx_local_info(${_msg})

  if(${variable}_RESULT)
    foreach(definition ${${variable}_DEFINITIONS})
      hpx_local_add_config_define(${definition})
    endforeach()
  elseif(${variable}_REQUIRED)
    hpx_local_warn("Test failed, detailed output:\n\n${${variable}_OUTPUT}")
    hpx_local_error(${${variable}_REQUIRED})
  endif()
endfunction()

# ##############################################################################
function(hpx_local_cpuid target variable)
  hpx_local_add_config_test(
    ${variable}
    SOURCE cmake/tests/cpuid.cpp
    COMPILE_DEFINITIONS "${boost_include_dir}" "${include_dir}"
    FILE EXECUTE
    ARGS "${target}" ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_local_check_for_unistd_h)
  hpx_local_add_config_test(
    HPXLocal_WITH_UNISTD_H
    SOURCE cmake/tests/unistd_h.cpp
    FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_local_check_for_libfun_std_experimental_optional)
  hpx_local_add_config_test(
    HPXLocal_WITH_LIBFUN_EXPERIMENTAL_OPTIONAL
    SOURCE cmake/tests/libfun_std_experimental_optional.cpp
    FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_local_check_for_cxx11_std_atomic)
  # Make sure HPXLocal_HAVE_LIBATOMIC is removed from the cache if necessary
  if(NOT HPXLocal_WITH_CXX11_ATOMIC)
    unset(HPXLocal_CXX11_STD_ATOMIC_LIBRARIES CACHE)
  endif()

  # first see if we can build atomics with no -latomics
  hpx_local_add_config_test(
    HPXLocal_WITH_CXX11_ATOMIC
    SOURCE cmake/tests/cxx11_std_atomic.cpp
    LIBRARIES ${HPXLocal_CXX11_STD_ATOMIC_LIBRARIES}
    FILE ${ARGN}
  )

  if(NOT MSVC)
    # Sometimes linking against libatomic is required, if the platform doesn't
    # support lock-free atomics. We already know that MSVC works
    if(NOT HPXLocal_WITH_CXX11_ATOMIC)
      set(HPXLocal_CXX11_STD_ATOMIC_LIBRARIES
          atomic
          CACHE STRING "std::atomics need separate library" FORCE
      )
      unset(HPXLocal_WITH_CXX11_ATOMIC CACHE)
      hpx_local_add_config_test(
        HPXLocal_WITH_CXX11_ATOMIC
        SOURCE cmake/tests/cxx11_std_atomic.cpp
        LIBRARIES ${HPXLocal_CXX11_STD_ATOMIC_LIBRARIES}
        FILE ${ARGN}
      )
      if(NOT HPXLocal_WITH_CXX11_ATOMIC)
        unset(HPXLocal_CXX11_STD_ATOMIC_LIBRARIES CACHE)
        unset(HPXLocal_WITH_CXX11_ATOMIC CACHE)
      endif()
    endif()
  endif()
endfunction()

# Separately check for 128 bit atomics
function(hpx_local_check_for_cxx11_std_atomic_128bit)
  hpx_local_add_config_test(
    HPXLocal_WITH_CXX11_ATOMIC_128BIT
    SOURCE cmake/tests/cxx11_std_atomic_128bit.cpp
    LIBRARIES ${HPXLocal_CXX11_STD_ATOMIC_LIBRARIES}
    FILE ${ARGN}
  )
  if(NOT MSVC)
    # Sometimes linking against libatomic is required, if the platform doesn't
    # support lock-free atomics. We already know that MSVC works
    if(NOT HPXLocal_WITH_CXX11_ATOMIC_128BIT)
      set(HPXLocal_CXX11_STD_ATOMIC_LIBRARIES
          atomic
          CACHE STRING "std::atomics need separate library" FORCE
      )
      unset(HPXLocal_WITH_CXX11_ATOMIC_128BIT CACHE)
      hpx_local_add_config_test(
        HPXLocal_WITH_CXX11_ATOMIC_128BIT
        SOURCE cmake/tests/cxx11_std_atomic_128bit.cpp
        LIBRARIES ${HPXLocal_CXX11_STD_ATOMIC_LIBRARIES}
        FILE ${ARGN}
      )
      if(NOT HPXLocal_WITH_CXX11_ATOMIC_128BIT)
        # Adding -latomic did not help, so we don't attempt to link to it later
        unset(HPXLocal_CXX11_STD_ATOMIC_LIBRARIES CACHE)
        unset(HPXLocal_WITH_CXX11_ATOMIC_128BIT CACHE)
      endif()
    endif()
  endif()
endfunction()

# ##############################################################################
function(hpx_local_check_for_cxx11_std_shared_ptr_lwg3018)
  hpx_local_add_config_test(
    HPXLocal_WITH_CXX11_SHARED_PTR_LWG3018
    SOURCE cmake/tests/cxx11_std_shared_ptr_lwg3018.cpp
    FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_local_check_for_c11_aligned_alloc)
  hpx_local_add_config_test(
    HPXLocal_WITH_C11_ALIGNED_ALLOC
    SOURCE cmake/tests/c11_aligned_alloc.cpp
    FILE ${ARGN}
  )
endfunction()

function(hpx_local_check_for_cxx17_std_aligned_alloc)
  hpx_local_add_config_test(
    HPXLocal_WITH_CXX17_STD_ALIGNED_ALLOC
    SOURCE cmake/tests/cxx17_std_aligned_alloc.cpp
    FILE ${ARGN}
  )
endfunction()

function(hpx_local_check_for_cxx17_std_execution_policies)
  hpx_local_add_config_test(
    HPXLocal_WITH_CXX17_STD_EXECUTION_POLICES
    SOURCE cmake/tests/cxx17_std_execution_policies.cpp
    FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_local_check_for_cxx11_std_quick_exit)
  hpx_local_add_config_test(
    HPXLocal_WITH_CXX11_STD_QUICK_EXIT
    SOURCE cmake/tests/cxx11_std_quick_exit.cpp
    FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_local_check_for_cxx17_aligned_new)
  hpx_local_add_config_test(
    HPXLocal_WITH_CXX17_ALIGNED_NEW
    SOURCE cmake/tests/cxx17_aligned_new.cpp
    FILE ${ARGN}
    REQUIRED
  )
endfunction()

# ##############################################################################
function(hpx_local_check_for_cxx17_filesystem)
  hpx_local_add_config_test(
    HPXLocal_WITH_CXX17_FILESYSTEM
    SOURCE cmake/tests/cxx17_filesystem.cpp
    FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_local_check_for_cxx17_hardware_destructive_interference_size)
  hpx_local_add_config_test(
    HPXLocal_WITH_CXX17_HARDWARE_DESTRUCTIVE_INTERFERENCE_SIZE
    SOURCE cmake/tests/cxx17_hardware_destructive_interference_size.cpp
    FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_local_check_for_cxx17_std_transform_scan)
  hpx_local_add_config_test(
    HPXLocal_WITH_CXX17_STD_TRANSFORM_SCAN_ALGORITHMS
    SOURCE cmake/tests/cxx17_std_transform_scan_algorithms.cpp
    FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_local_check_for_cxx17_std_scan)
  hpx_local_add_config_test(
    HPXLocal_WITH_CXX17_STD_SCAN_ALGORITHMS
    SOURCE cmake/tests/cxx17_std_scan_algorithms.cpp
    FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_local_check_for_cxx17_shared_ptr_array)
  hpx_local_add_config_test(
    HPXLocal_WITH_CXX17_SHARED_PTR_ARRAY
    SOURCE cmake/tests/cxx17_shared_ptr_array.cpp
    FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_local_check_for_cxx17_copy_elision)
  hpx_local_add_config_test(
    HPXLocal_WITH_CXX17_COPY_ELISION
    SOURCE cmake/tests/cxx17_copy_elision.cpp
    FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_local_check_for_cxx20_coroutines)
  hpx_local_add_config_test(
    HPXLocal_WITH_CXX20_COROUTINES
    SOURCE cmake/tests/cxx20_coroutines.cpp
    FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_local_check_for_cxx20_lambda_capture)
  hpx_local_add_config_test(
    HPXLocal_WITH_CXX20_LAMBDA_CAPTURE
    SOURCE cmake/tests/cxx20_lambda_capture.cpp
    FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_local_check_for_cxx20_perfect_pack_capture)
  hpx_local_add_config_test(
    HPXLocal_WITH_CXX20_PERFECT_PACK_CAPTURE
    SOURCE cmake/tests/cxx20_perfect_pack_capture.cpp
    FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_local_check_for_cxx20_experimental_simd)
  hpx_local_add_config_test(
    HPXLocal_WITH_CXX20_EXPERIMENTAL_SIMD
    SOURCE cmake/tests/cxx20_experimental_simd.cpp
    FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_local_check_for_cxx20_no_unique_address_attribute)
  hpx_local_add_config_test(
    HPXLocal_WITH_CXX20_NO_UNIQUE_ADDRESS_ATTRIBUTE
    SOURCE cmake/tests/cxx20_no_unique_address_attribute.cpp
    FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_local_check_for_cxx20_paren_initialization_of_aggregates)
  hpx_local_add_config_test(
    HPXLocal_WITH_CXX20_PAREN_INITIALIZATION_OF_AGGREGATES
    SOURCE cmake/tests/cxx20_paren_initialization_of_aggregates.cpp
    FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_local_check_for_cxx20_std_disable_sized_sentinel_for)
  hpx_local_add_config_test(
    HPXLocal_WITH_CXX20_STD_DISABLE_SIZED_SENTINEL_FOR
    SOURCE cmake/tests/cxx20_std_disable_sized_sentinel_for.cpp
    FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_local_check_for_cxx20_std_endian)
  hpx_local_add_config_test(
    HPXLocal_WITH_CXX20_STD_ENDIAN
    SOURCE cmake/tests/cxx20_std_endian.cpp
    FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_local_check_for_cxx20_std_execution_policies)
  hpx_local_add_config_test(
    HPXLocal_WITH_CXX20_STD_EXECUTION_POLICES
    SOURCE cmake/tests/cxx20_std_execution_policies.cpp
    FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_local_check_for_cxx20_std_ranges_iter_swap)
  hpx_local_add_config_test(
    HPXLocal_WITH_CXX20_STD_RANGES_ITER_SWAP
    SOURCE cmake/tests/cxx20_std_ranges_iter_swap.cpp
    FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_local_check_for_builtin_integer_pack)
  hpx_local_add_config_test(
    HPXLocal_WITH_BUILTIN_INTEGER_PACK
    SOURCE cmake/tests/builtin_integer_pack.cpp
    FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_local_check_for_builtin_make_integer_seq)
  hpx_local_add_config_test(
    HPXLocal_WITH_BUILTIN_MAKE_INTEGER_SEQ
    SOURCE cmake/tests/builtin_make_integer_seq.cpp
    FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_local_check_for_builtin_make_integer_seq_cuda)
  hpx_local_add_config_test(
    HPXLocal_WITH_BUILTIN_MAKE_INTEGER_SEQ_CUDA
    SOURCE cmake/tests/builtin_make_integer_seq.cu CUDA
    FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_local_check_for_builtin_type_pack_element)
  hpx_local_add_config_test(
    HPXLocal_WITH_BUILTIN_TYPE_PACK_ELEMENT
    SOURCE cmake/tests/builtin_type_pack_element.cpp
    FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_local_check_for_builtin_type_pack_element_cuda)
  hpx_local_add_config_test(
    HPXLocal_WITH_BUILTIN_TYPE_PACK_ELEMENT_CUDA
    SOURCE cmake/tests/builtin_type_pack_element.cu CUDA
    FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_local_check_for_mm_prefetch)
  hpx_local_add_config_test(
    HPXLocal_WITH_MM_PREFETCH
    SOURCE cmake/tests/mm_prefetch.cpp
    FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_local_check_for_stable_inplace_merge)
  hpx_local_add_config_test(
    HPXLocal_WITH_STABLE_INPLACE_MERGE
    SOURCE cmake/tests/stable_inplace_merge.cpp
    FILE ${ARGN}
  )
endfunction()
