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

set(HPX_ADDCONFIGTEST_LOADED TRUE)

include(CheckLibraryExists)

function(add_hpx_config_test variable)
  set(options FILE EXECUTE)
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
  if(NOT DEFINED ${variable})
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
      set(test_source
          "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/config_tests/${variable_lc}.cpp"
      )
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
      HPX_TARGET_COMPILE_OPTIONS_PUBLIC_VAR GLOBAL
      PROPERTY HPX_TARGET_COMPILE_OPTIONS_PUBLIC
    )
    get_property(
      HPX_TARGET_COMPILE_OPTIONS_PRIVATE_VAR GLOBAL
      PROPERTY HPX_TARGET_COMPILE_OPTIONS_PRIVATE
    )
    set(HPX_TARGET_COMPILE_OPTIONS_VAR
        ${HPX_TARGET_COMPILE_OPTIONS_PUBLIC_VAR}
        ${HPX_TARGET_COMPILE_OPTIONS_PRIVATE_VAR}
    )
    foreach(_flag ${HPX_TARGET_COMPILE_OPTIONS_VAR})
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

    get_property(
      _base_libraries
      TARGET hpx_base_libraries
      PROPERTY INTERFACE_LINK_LIBRARIES
    )
    set(CONFIG_TEST_LINK_LIBRARIES ${_base_libraries} ${${variable}_LIBRARIES})

    if(${variable}_EXECUTE)
      if(NOT CMAKE_CROSSCOMPILING)
        # cmake-format: off
        try_run(
          ${variable}_RUN_RESULT ${variable}_COMPILE_RESULT
          ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/config_tests
          ${test_source}
          CMAKE_FLAGS
            "-DINCLUDE_DIRECTORIES=${CONFIG_TEST_INCLUDE_DIRS}"
            "-DLINK_DIRECTORIES=${CONFIG_TEST_LINK_DIRS}"
            "-DLINK_LIBRARIES=${CONFIG_TEST_LINK_LIBRARIES}"
            "-DCOMPILE_DEFINITIONS=${CONFIG_TEST_COMPILE_DEFINITIONS}"
          CXX_STANDARD ${HPX_CXX_STANDARD}
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
      # cmake-format: off
      try_compile(
        ${variable}_RESULT
        ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/config_tests
        ${test_source}
        CMAKE_FLAGS
          "-DINCLUDE_DIRECTORIES=${CONFIG_TEST_INCLUDE_DIRS}"
          "-DLINK_DIRECTORIES=${CONFIG_TEST_LINK_DIRS}"
          "-DLINK_LIBRARIES=${CONFIG_TEST_LINK_LIBRARIES}"
          "-DCOMPILE_DEFINITIONS=${CONFIG_TEST_COMPILE_DEFINITIONS}"
        OUTPUT_VARIABLE ${variable}_OUTPUT
        CXX_STANDARD ${HPX_CXX_STANDARD}
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS FALSE
        COPY_FILE ${test_binary}
      )
      # cmake-format: on
      hpx_debug("Compile test: ${variable}")
      hpx_debug("Compilation output: ${${variable}_OUTPUT}")
    endif()

    set(_run_msg "Success")
  else()
    set(${variable}_RESULT ${${variable}})
    if(NOT _run_msg)
      set(_run_msg "pre-set to ${${variable}}")
    endif()
  endif()

  string(TOUPPER "${variable}" variable_uc)
  set(_msg "Performing Test ${variable_uc}")

  if(${variable}_RESULT)
    set(_msg "${_msg} - ${_run_msg}")
  else()
    set(_msg "${_msg} - Failed")
  endif()

  set(${variable}
      ${${variable}_RESULT}
      CACHE INTERNAL ""
  )
  hpx_info(${_msg})

  if(${variable}_RESULT)
    foreach(definition ${${variable}_DEFINITIONS})
      hpx_add_config_define(${definition})
    endforeach()
  elseif(${variable}_REQUIRED)
    hpx_warn("Test failed, detailed output:\n\n${${variable}_OUTPUT}")
    hpx_error(${${variable}_REQUIRED})
  endif()
endfunction()

# Makes it possible to provide a feature test that is able to test the compiler
# to build parts of HPX directly when the given definition is defined.
function(add_hpx_in_framework_config_test variable)
  # Generate the config only if the test wasn't executed yet
  if(NOT DEFINED ${variable})
    # Location to generate the config headers to
    set(${variable}_GENERATED_DIR
        "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/config_tests/header-${variable}"
    )
    generate_config_defines_header(${${variable}_GENERATED_DIR})
  endif()

  set(options)
  set(one_value_args)
  set(multi_value_args DEFINITIONS INCLUDE_DIRECTORIES COMPILE_DEFINITIONS)
  cmake_parse_arguments(
    ${variable} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN}
  )

  # We call the generic feature test method while modifying some existing parsed
  # arguments in order to alter the INCLUDE_DIRECTORIES and the
  # COMPILE_DEFINITIONS. It's important here not to link the config test against
  # an executable because otherwise this will result in unresolved references to
  # the HPX library, that wasn't built as of now.
  add_hpx_config_test(
    ${variable} ${${variable}_UNPARSED_ARGUMENTS}
    DEFINITIONS ${${variable}_DEFINITIONS}
    COMPILE_DEFINITIONS
      ${${variable}_COMPILE_DEFINITIONS}
      # We add the definitions we test to the existing compile definitions.
      ${${variable}_DEFINITIONS}
      # Add HPX_NO_VERSION_CHECK to make header only parts of HPX available
      # without requiring to link against the HPX sources. We can remove this
      # workaround as soon as CMake 3.6 is the minimal required version and
      # supports: CMAKE_TRY_COMPILE_TARGET_TYPE = STATIC_LIBRARY when using
      # try_compile to not to throw errors on unresolved symbols.
      HPX_NO_VERSION_CHECK
    INCLUDE_DIRECTORIES
      ${${variable}_INCLUDE_DIRECTORIES}
      # We add the generated headers to the include dirs
      ${${variable}_GENERATED_DIR}
  )

  if(DEFINED ${variable}_GENERATED_DIR)
    # Cleanup the generated header
    file(REMOVE_RECURSE "${${variable}_GENERATED_DIR}")
  endif()
endfunction()

# ##############################################################################
function(hpx_cpuid target variable)
  add_hpx_config_test(
    ${variable}
    SOURCE cmake/tests/cpuid.cpp
    COMPILE_DEFINITIONS "${boost_include_dir}" "${include_dir}" FILE EXECUTE
    ARGS "${target}" ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_check_for_unistd_h)
  add_hpx_config_test(
    HPX_WITH_UNISTD_H SOURCE cmake/tests/unistd_h.cpp FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_check_for_libfun_std_experimental_optional)
  add_hpx_config_test(
    HPX_WITH_LIBFUN_EXPERIMENTAL_OPTIONAL
    SOURCE cmake/tests/libfun_std_experimental_optional.cpp FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_check_for_cxx11_std_atomic)
  # Make sure HPX_HAVE_LIBATOMIC is removed from the cache if necessary
  if(NOT HPX_WITH_CXX11_ATOMIC)
    unset(HPX_HAVE_LIBATOMIC CACHE)
  endif()

  if(NOT MSVC)
    # Sometimes linking against libatomic is required for atomic ops, if the
    # platform doesn't support lock-free atomics. We know, it's not needed for
    # MSVC
    check_library_exists(atomic __atomic_fetch_add_4 "" HPX_HAVE_LIBATOMIC)
    if(HPX_HAVE_LIBATOMIC)
      set(HPX_CXX11_STD_ATOMIC_LIBRARIES
          atomic
          CACHE BOOL "std::atomics need separate library" FORCE
      )
    endif()
  endif()

  add_hpx_config_test(
    HPX_WITH_CXX11_ATOMIC
    SOURCE cmake/tests/cxx11_std_atomic.cpp
    LIBRARIES ${HPX_CXX11_STD_ATOMIC_LIBRARIES} FILE ${ARGN}
  )
endfunction()

# Separately check for 128 bit atomics
function(hpx_check_for_cxx11_std_atomic_128bit)
  add_hpx_config_test(
    HPX_WITH_CXX11_ATOMIC_128BIT
    SOURCE cmake/tests/cxx11_std_atomic_128bit.cpp
    LIBRARIES ${HPX_CXX11_STD_ATOMIC_LIBRARIES} FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_check_for_cxx11_std_shared_ptr_lwg3018)
  add_hpx_config_test(
    HPX_WITH_CXX11_SHARED_PTR_LWG3018
    SOURCE cmake/tests/cxx11_std_shared_ptr_lwg3018.cpp FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_check_for_c11_aligned_alloc)
  add_hpx_config_test(
    HPX_WITH_C11_ALIGNED_ALLOC
    SOURCE cmake/tests/c11_aligned_alloc.cpp FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_check_for_cxx11_std_quick_exit)
  add_hpx_config_test(
    HPX_WITH_CXX11_STD_QUICK_EXIT SOURCE cmake/tests/cxx11_std_quick_exit.cpp
                                         FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_check_for_cxx17_aligned_new)
  add_hpx_config_test(
    HPX_WITH_CXX17_ALIGNED_NEW
    SOURCE cmake/tests/cxx17_aligned_new.cpp FILE ${ARGN}
    REQUIRED
  )
endfunction()

# ##############################################################################
function(hpx_check_for_cxx17_filesystem)
  add_hpx_config_test(
    HPX_WITH_CXX17_FILESYSTEM SOURCE cmake/tests/cxx17_filesystem.cpp FILE
                                     ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_check_for_cxx17_fold_expressions)
  add_hpx_config_test(
    HPX_WITH_CXX17_FOLD_EXPRESSIONS
    SOURCE cmake/tests/cxx17_fold_expressions.cpp FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_check_for_cxx17_fallthrough_attribute)
  add_hpx_config_test(
    HPX_WITH_CXX17_FALLTHROUGH_ATTRIBUTE
    SOURCE cmake/tests/cxx17_fallthrough_attribute.cpp FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_check_for_cxx17_nodiscard_attribute)
  add_hpx_config_test(
    HPX_WITH_CXX17_NODISCARD_ATTRIBUTE
    SOURCE cmake/tests/cxx17_nodiscard_attribute.cpp FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_check_for_cxx17_hardware_destructive_interference_size)
  add_hpx_config_test(
    HPX_WITH_CXX17_HARDWARE_DESTRUCTIVE_INTERFERENCE_SIZE
    SOURCE cmake/tests/cxx17_hardware_destructive_interference_size.cpp FILE
           ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_check_for_cxx17_std_in_place_type_t)
  add_hpx_config_test(
    HPX_WITH_CXX17_STD_IN_PLACE_TYPE_T
    SOURCE cmake/tests/cxx17_std_in_place_type_t.cpp FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_check_for_cxx17_maybe_unused)
  add_hpx_config_test(
    HPX_WITH_CXX17_MAYBE_UNUSED SOURCE cmake/tests/cxx17_maybe_unused.cpp FILE
                                       ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_check_for_cxx17_deduction_guides)
  add_hpx_config_test(
    HPX_WITH_CXX17_DEDUCTION_GUIDES
    SOURCE cmake/tests/cxx17_deduction_guides.cpp FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_check_for_cxx17_structured_bindings)
  add_hpx_config_test(
    HPX_WITH_CXX17_STRUCTURED_BINDINGS
    SOURCE cmake/tests/cxx17_structured_bindings.cpp FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_check_for_cxx17_if_constexpr)
  add_hpx_config_test(
    HPX_WITH_CXX17_IF_CONSTEXPR SOURCE cmake/tests/cxx17_if_constexpr.cpp FILE
                                       ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_check_for_cxx17_inline_constexpr_variable)
  add_hpx_config_test(
    HPX_WITH_CXX17_INLINE_CONSTEXPR_VALUE
    SOURCE cmake/tests/cxx17_inline_constexpr_variable.cpp FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_check_for_cxx17_noexcept_functions_as_nontype_template_arguments)
  add_hpx_config_test(
    HPX_WITH_CXX17_NOEXCEPT_FUNCTIONS_AS_NONTYPE_TEMPLATE_ARGUMENTS
    SOURCE cmake/tests/cxx17_noexcept_function.cpp FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_check_for_cxx17_std_variant)
  add_hpx_config_test(
    HPX_WITH_CXX17_STD_VARIANT SOURCE cmake/tests/cxx17_std_variant.cpp FILE
                                      ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_check_for_cxx17_std_transform_scan)
  add_hpx_config_test(
    HPX_WITH_CXX17_STD_TRANSFORM_SCAN_ALGORITHMS
    SOURCE cmake/tests/cxx17_std_transform_scan_algorithms.cpp FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_check_for_cxx17_std_scan)
  add_hpx_config_test(
    HPX_WITH_CXX17_STD_SCAN_ALGORITHMS
    SOURCE cmake/tests/cxx17_std_scan_algorithms.cpp FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_check_for_cxx17_shared_ptr_array)
  add_hpx_config_test(
    HPX_WITH_CXX17_SHARED_PTR_ARRAY
    SOURCE cmake/tests/cxx17_shared_ptr_array.cpp FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_check_for_cxx17_std_nontype_template_parameter_auto)
  add_hpx_config_test(
    HPX_WITH_CXX17_NONTYPE_TEMPLATE_PARAMETER_AUTO
    SOURCE cmake/tests/cxx17_std_nontype_template_parameter_auto.cpp FILE
           ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_check_for_cxx20_coroutines)
  add_hpx_config_test(
    HPX_WITH_CXX20_COROUTINES SOURCE cmake/tests/cxx20_coroutines.cpp FILE
                                     ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_check_for_cxx20_std_disable_sized_sentinel_for)
  add_hpx_config_test(
    HPX_WITH_CXX20_STD_DISABLE_SIZED_SENTINEL_FOR
    SOURCE cmake/tests/cxx20_std_disable_sized_sentinel_for.cpp FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_check_for_cxx20_no_unique_address_attribute)
  add_hpx_config_test(
    HPX_WITH_CXX20_NO_UNIQUE_ADDRESS_ATTRIBUTE
    SOURCE cmake/tests/cxx20_no_unique_address_attribute.cpp FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_check_for_builtin_integer_pack)
  add_hpx_config_test(
    HPX_WITH_BUILTIN_INTEGER_PACK SOURCE cmake/tests/builtin_integer_pack.cpp
                                         FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_check_for_builtin_make_integer_seq)
  add_hpx_config_test(
    HPX_WITH_BUILTIN_MAKE_INTEGER_SEQ
    SOURCE cmake/tests/builtin_make_integer_seq.cpp FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_check_for_builtin_type_pack_element)
  add_hpx_config_test(
    HPX_WITH_BUILTIN_TYPE_PACK_ELEMENT
    SOURCE cmake/tests/builtin_type_pack_element.cpp FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_check_for_mm_prefetch)
  add_hpx_config_test(
    HPX_WITH_MM_PREFETCH SOURCE cmake/tests/mm_prefetch.cpp FILE ${ARGN}
  )
endfunction()

# ##############################################################################
function(hpx_check_for_stable_inplace_merge)
  add_hpx_config_test(
    HPX_WITH_STABLE_INPLACE_MERGE SOURCE cmake/tests/stable_inplace_merge.cpp
                                         FILE ${ARGN}
  )
endfunction()
