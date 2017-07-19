# Copyright (c) 2011 Bryce Lelbach
# Copyright (c) 2014 Thomas Heller
# Copyright (c) 2017 Denis Blank
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_ADDCONFIGTEST_LOADED TRUE)

macro(add_hpx_config_test variable)
  set(options FILE EXECUTE)
  set(one_value_args SOURCE ROOT CMAKECXXFEATURE)
  set(multi_value_args INCLUDE_DIRECTORIES LINK_DIRECTORIES COMPILE_DEFINITIONS LIBRARIES ARGS DEFINITIONS REQUIRED)
  cmake_parse_arguments(${variable} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  set(_run_msg)
  if(${variable}_CMAKECXXFEATURE)
    # We don't have to run our own feasture test if there is a corresponding
    # cmake feature test and cmake reports the feature is supported on this
    # platform.
    list(FIND CMAKE_CXX_COMPILE_FEATURES ${${variable}_CMAKECXXFEATURE} __pos)
    if(NOT ${__pos} EQUAL -1)
      set(${variable} TRUE)
      set(_run_msg "Success (cmake feature test)")
    endif()
  endif()

  if(NOT DEFINED ${variable})
    file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/config_tests")

    string(TOLOWER "${variable}" variable_lc)
    if(${variable}_FILE)
      if(${variable}_ROOT)
        set(test_source "${${variable}_ROOT}/share/hpx-${HPX_VERSION}/${${variable}_SOURCE}")
      else()
        set(test_source "${PROJECT_SOURCE_DIR}/${${variable}_SOURCE}")
      endif()
    else()
      set(test_source
          "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/config_tests/${variable_lc}.cpp")
      file(WRITE "${test_source}"
           "${${variable}_SOURCE}\n")
    endif()
    set(test_binary ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/config_tests/${variable_lc})

    get_directory_property(CONFIG_TEST_INCLUDE_DIRS INCLUDE_DIRECTORIES)
    get_directory_property(CONFIG_TEST_LINK_DIRS LINK_DIRECTORIES)
    set(COMPILE_DEFINITIONS_TMP)
    set(CONFIG_TEST_COMPILE_DEFINITIONS)
    get_directory_property(COMPILE_DEFINITIONS_TMP COMPILE_DEFINITIONS)
    foreach(def IN LISTS COMPILE_DEFINITIONS_TMP ${variable}_COMPILE_DEFINITIONS)
      set(CONFIG_TEST_COMPILE_DEFINITIONS "${CONFIG_TEST_COMPILE_DEFINITIONS} -D${def}")
    endforeach()
    get_property(HPX_TARGET_COMPILE_OPTIONS_VAR GLOBAL PROPERTY HPX_TARGET_COMPILE_OPTIONS)
    foreach(_flag ${HPX_TARGET_COMPILE_OPTIONS_VAR})
      if(NOT "${_flag}" MATCHES "^\\$.*")
        set(CONFIG_TEST_COMPILE_DEFINITIONS "${CONFIG_TEST_COMPILE_DEFINITIONS} ${_flag}")
      endif()
    endforeach()

    set(CONFIG_TEST_INCLUDE_DIRS ${CONFIG_TEST_INCLUDE_DIRS} ${${variable}_INCLUDE_DIRECTORIES})
    set(CONFIG_TEST_LINK_DIRS ${CONFIG_TEST_LINK_DIRS} ${${variable}_LINK_DIRECTORIES})

    set(CONFIG_TEST_LINK_LIBRARIES ${HPX_LIBRARIES} ${${variable}_LIBRARIES})

    if(${variable}_EXECUTE)
      if(NOT CMAKE_CROSSCOMPILING)
        try_run(${variable}_RUN_RESULT ${variable}_COMPILE_RESULT
          ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/config_tests
          ${test_source}
          CMAKE_FLAGS
            "-DINCLUDE_DIRECTORIES=${CONFIG_TEST_INCLUDE_DIRS}"
            "-DLINK_DIRECTORIES=${CONFIG_TEST_LINK_DIRS}"
            "-DLINK_LIBRARIES=${CONFIG_TEST_LINK_LIBRARIES}"
            "-DCOMPILE_DEFINITIONS=${CONFIG_TEST_COMPILE_DEFINITIONS}"
          RUN_OUTPUT_VARIABLE ${variable}_OUTPUT
          ARGS ${${variable}_ARGS})
        if(${variable}_COMPILE_RESULT AND NOT ${variable}_RUN_RESULT)
          set(${variable}_RESULT TRUE)
        else()
          set(${variable}_RESULT FALSE)
        endif()
      else()
        set(${variable}_RESULT FALSE)
      endif()
    else()
      try_compile(${variable}_RESULT
        ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/config_tests
        ${test_source}
        CMAKE_FLAGS
          "-DINCLUDE_DIRECTORIES=${CONFIG_TEST_INCLUDE_DIRS}"
          "-DLINK_DIRECTORIES=${CONFIG_TEST_LINK_DIRS}"
          "-DLINK_LIBRARIES=${CONFIG_TEST_LINK_LIBRARIES}"
          "-DCOMPILE_DEFINITIONS=${CONFIG_TEST_COMPILE_DEFINITIONS}"
        OUTPUT_VARIABLE ${variable}_OUTPUT
        COPY_FILE ${test_binary})
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

  set(${variable} ${${variable}_RESULT} CACHE INTERNAL "")
  hpx_info(${_msg})

  if(${variable}_RESULT)
    foreach(definition ${${variable}_DEFINITIONS})
      hpx_add_config_define(${definition})
    endforeach()
  elseif(${variable}_REQUIRED)
    hpx_warn("Test failed, detailed output:\n\n${${variable}_OUTPUT}")
    hpx_error(${${variable}_REQUIRED})
  endif()
endmacro()

# Makes it possible to provide a feature test that is able to
# test the compiler to build parts of HPX directly when the given definition
# is defined.
macro(add_hpx_in_framework_config_test variable)
  # Generate the config only if the test wasn't executed yet
  if(NOT DEFINED ${variable})
    # Location to generate the config headers to
    set(${variable}_GENERATED_DIR
      "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/config_tests/header-${variable}")
    generate_config_defines_header(${${variable}_GENERATED_DIR})
  endif()

  set(options)
  set(one_value_args)
  set(multi_value_args DEFINITIONS INCLUDE_DIRECTORIES COMPILE_DEFINITIONS)
  cmake_parse_arguments(${variable} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  # We call the generic feature test method while modifying some
  # existing parsed arguments in order to alter the INCLUDE_DIRECTORIES
  # and the COMPILE_DEFINITIONS.
  # It's important here not to link the config test against an executable
  # because otherwise this will result in unresolved references to the
  # HPX library, that wasn't built as of now.
  add_hpx_config_test(${variable}
                      ${${variable}_UNPARSED_ARGUMENTS}
                      DEFINITIONS
                        ${${variable}_DEFINITIONS}
                      COMPILE_DEFINITIONS
                        ${${variable}_COMPILE_DEFINITIONS}
                        # We add the definitions we test to the
                        # existing compile definitions.
                        ${${variable}_DEFINITIONS}
                        # Add HPX_NO_VERSION_CHECK to make header only
                        # parts of HPX available without requiring to link
                        # against the HPX sources.
                        # We can remove this workaround as soon as CMake 3.6
                        # is the minimal required version and supports:
                        # CMAKE_TRY_COMPILE_TARGET_TYPE = STATIC_LIBRARY
                        # when using try_compile to not to throw errors
                        # on unresolved symbols.
                        HPX_NO_VERSION_CHECK
                      INCLUDE_DIRECTORIES
                        ${${variable}_INCLUDE_DIRECTORIES}
                        # We add the generated headers to the include dirs
                        ${${variable}_GENERATED_DIR})

  if(DEFINED ${variable}_GENERATED_DIR)
    # Cleanup the generated header
    file(REMOVE_RECURSE "${${variable}_GENERATED_DIR}")
  endif()
endmacro()

###############################################################################
macro(hpx_cpuid target variable)
  add_hpx_config_test(${variable}
    SOURCE cmake/tests/cpuid.cpp
    FLAGS "${boost_include_dir}" "${include_dir}"
    FILE EXECUTE ARGS "${target}" ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_unistd_h)
  add_hpx_config_test(HPX_WITH_UNISTD_H
    SOURCE cmake/tests/unistd_h.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_alias_templates)
  add_hpx_config_test(HPX_WITH_CXX11_ALIAS_TEMPLATES
    SOURCE cmake/tests/cxx11_alias_templates.cpp
    FILE ${ARGN}
    CMAKECXXFEATURE cxx_alias_templates)
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_auto)
  add_hpx_config_test(HPX_WITH_CXX11_AUTO
    SOURCE cmake/tests/cxx11_auto.cpp
    FILE ${ARGN}
    CMAKECXXFEATURE cxx_auto_type)
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_constexpr)
  add_hpx_config_test(HPX_WITH_CXX11_CONSTEXPR
    SOURCE cmake/tests/cxx11_constexpr.cpp
    FILE ${ARGN}
    CMAKECXXFEATURE cxx_constexpr)
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_decltype)
  add_hpx_config_test(HPX_WITH_CXX11_DECLTYPE
    SOURCE cmake/tests/cxx11_decltype.cpp
    FILE ${ARGN}
    CMAKECXXFEATURE cxx_decltype)
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_sfinae_expression)
  add_hpx_config_test(HPX_WITH_CXX11_SFINAE_EXPRESSION
    SOURCE cmake/tests/cxx11_sfinae_expression.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_sfinae_expression_complete)
  add_hpx_in_framework_config_test(HPX_HAVE_CXX11_SFINAE_EXPRESSION_COMPLETE
    SOURCE cmake/tests/cxx11_sfinae_expression_complete.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_defaulted_functions)
  add_hpx_config_test(HPX_WITH_CXX11_DEFAULTED_FUNCTIONS
    SOURCE cmake/tests/cxx11_defaulted_functions.cpp
    FILE ${ARGN}
    CMAKECXXFEATURE cxx_defaulted_functions)
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_deleted_functions)
  add_hpx_config_test(HPX_WITH_CXX11_DELETED_FUNCTIONS
    SOURCE cmake/tests/cxx11_deleted_functions.cpp
    FILE ${ARGN}
    CMAKECXXFEATURE cxx_deleted_functions)
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_explicit_cvt_ops)
  add_hpx_config_test(HPX_WITH_CXX11_EXPLICIT_CONVERSION_OPERATORS
    SOURCE cmake/tests/cxx11_explicit_cvt_ops.cpp
    FILE ${ARGN}
    CMAKECXXFEATURE cxx_explicit_conversions)
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_explicit_variadic_templates)
  add_hpx_config_test(HPX_WITH_CXX11_EXPLICIT_VARIADIC_TEMPLATES
    SOURCE cmake/tests/cxx11_explicit_variadic_templates.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_extended_friend_declarations)
  add_hpx_config_test(HPX_WITH_CXX11_EXTENDED_FRIEND_DECLARATIONS
    SOURCE cmake/tests/cxx11_extended_friend_declarations.cpp
    FILE ${ARGN}
    CMAKECXXFEATURE cxx_extended_friend_declarations)
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_function_template_default_args)
  add_hpx_config_test(HPX_WITH_CXX11_FUNCTION_TEMPLATE_DEFAULT_ARGS
    SOURCE cmake/tests/cxx11_function_template_default_args.cpp
    FILE ${ARGN}
    CMAKECXXFEATURE cxx_default_function_template_args)
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_inline_namespaces)
  add_hpx_config_test(HPX_WITH_CXX11_INLINE_NAMESPACES
    SOURCE cmake/tests/cxx11_inline_namespaces.cpp
    FILE ${ARGN}
    CMAKECXXFEATURE cxx_inline_namespaces)
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_lambdas)
  add_hpx_config_test(HPX_WITH_CXX11_LAMBDAS
    SOURCE cmake/tests/cxx11_lambdas.cpp
    FILE ${ARGN}
    CMAKECXXFEATURE cxx_lambdas)
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_noexcept)
  add_hpx_config_test(HPX_WITH_CXX11_NOEXCEPT
    SOURCE cmake/tests/cxx11_noexcept.cpp
    FILE ${ARGN}
    CMAKECXXFEATURE cxx_noexcept)
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_nsdmi)
  add_hpx_config_test(HPX_WITH_CXX11_NSDMI
    SOURCE cmake/tests/cxx11_non_static_data_member_initialization.cpp
    FILE ${ARGN}
    CMAKECXXFEATURE cxx_nonstatic_member_init)
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_nullptr)
  add_hpx_config_test(HPX_WITH_CXX11_NULLPTR
    SOURCE cmake/tests/cxx11_nullptr.cpp
    FILE ${ARGN}
    CMAKECXXFEATURE cxx_nullptr)
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_range_based_for)
  add_hpx_config_test(HPX_WITH_CXX11_RANGE_BASED_FOR
    SOURCE cmake/tests/cxx11_range_based_for.cpp
    FILE ${ARGN}
    CMAKECXXFEATURE cxx_range_for)
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_rvalue_references)
  add_hpx_config_test(HPX_WITH_CXX11_RVALUE_REFERENCES
    SOURCE cmake/tests/cxx11_rvalue_references.cpp
    FILE ${ARGN}
    CMAKECXXFEATURE cxx_rvalue_references)
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_scoped_enums)
  add_hpx_config_test(HPX_WITH_CXX11_SCOPED_ENUMS
    SOURCE cmake/tests/cxx11_scoped_enums.cpp
    FILE ${ARGN}
    CMAKECXXFEATURE cxx_strong_enums)
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_static_assert)
  add_hpx_config_test(HPX_WITH_CXX11_STATIC_ASSERT
    SOURCE cmake/tests/cxx11_static_assert.cpp
    FILE ${ARGN}
    CMAKECXXFEATURE cxx_static_assert)
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_variadic_macros)
  add_hpx_config_test(HPX_WITH_CXX11_VARIADIC_MACROS
    SOURCE cmake/tests/cxx11_variadic_macros.cpp
    FILE ${ARGN}
    CMAKECXXFEATURE cxx_variadic_macros)
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_variadic_templates)
  add_hpx_config_test(HPX_WITH_CXX11_VARIADIC_TEMPLATES
    SOURCE cmake/tests/cxx11_variadic_templates.cpp
    FILE ${ARGN}
    CMAKECXXFEATURE cxx_variadic_templates)
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_std_array)
  add_hpx_config_test(HPX_WITH_CXX11_ARRAY
    SOURCE cmake/tests/cxx11_std_array.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_std_chrono)
  add_hpx_config_test(HPX_WITH_CXX11_CHRONO
    SOURCE cmake/tests/cxx11_std_chrono.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_std_cstdint)
  add_hpx_config_test(HPX_WITH_CXX11_CSTDINT
    SOURCE cmake/tests/cxx11_std_cstdint.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_std_exception_ptr)
  add_hpx_config_test(HPX_WITH_CXX11_EXCEPTION_PTR
    SOURCE cmake/tests/cxx11_std_exception_ptr.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_std_initializer_list)
  add_hpx_config_test(HPX_WITH_CXX11_STD_INITIALIZER_LIST
    SOURCE cmake/tests/cxx11_std_initializer_list.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_std_is_bind_expression)
  add_hpx_config_test(HPX_WITH_CXX11_IS_BIND_EXPRESSION
    SOURCE cmake/tests/cxx11_std_is_bind_expression.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_std_is_placeholder)
  add_hpx_config_test(HPX_WITH_CXX11_IS_PLACEHOLDER
    SOURCE cmake/tests/cxx11_std_is_placeholder.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_std_is_trivially_copyable)
  add_hpx_config_test(HPX_WITH_CXX11_IS_TRIVIALLY_COPYABLE
    SOURCE cmake/tests/cxx11_std_is_trivially_copyable.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_std_lock_guard)
  add_hpx_config_test(HPX_WITH_CXX11_LOCK_GUARD
    SOURCE cmake/tests/cxx11_std_lock_guard.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_std_range_access)
  add_hpx_config_test(HPX_WITH_CXX11_RANGE_ACCESS
    SOURCE cmake/tests/cxx11_std_range_access.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_std_reference_wrapper)
  add_hpx_config_test(HPX_WITH_CXX11_REFERENCE_WRAPPER
    SOURCE cmake/tests/cxx11_std_reference_wrapper.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_std_shared_ptr)
  add_hpx_config_test(HPX_WITH_CXX11_SHARED_PTR
    SOURCE cmake/tests/cxx11_std_shared_ptr.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_std_shuffle)
  add_hpx_config_test(HPX_WITH_CXX11_SHUFFLE
    SOURCE cmake/tests/cxx11_std_shuffle.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_std_thread)
  add_hpx_config_test(HPX_WITH_CXX11_THREAD
    SOURCE cmake/tests/cxx11_std_thread.cpp
    LIBRARIES "-pthread"
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_std_to_string)
  add_hpx_config_test(HPX_WITH_CXX11_TO_STRING
    SOURCE cmake/tests/cxx11_std_to_string.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_std_unique_lock)
  add_hpx_config_test(HPX_WITH_CXX11_UNIQUE_LOCK
    SOURCE cmake/tests/cxx11_std_unique_lock.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_std_unique_ptr)
  add_hpx_config_test(HPX_WITH_CXX11_UNIQUE_PTR
    SOURCE cmake/tests/cxx11_std_unique_ptr.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_std_unordered_map)
  add_hpx_config_test(HPX_WITH_CXX11_UNORDERED_MAP
    SOURCE cmake/tests/cxx11_std_unordered_map.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_std_unordered_set)
  add_hpx_config_test(HPX_WITH_CXX11_UNORDERED_SET
    SOURCE cmake/tests/cxx11_std_unordered_set.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx14_constexpr)
  add_hpx_config_test(HPX_WITH_CXX14_CONSTEXPR
    SOURCE cmake/tests/cxx14_constexpr.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx14_lambdas)
  add_hpx_config_test(HPX_WITH_CXX14_LAMBDAS
    SOURCE cmake/tests/cxx14_lambdas.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx14_std_integer_sequence)
  add_hpx_config_test(HPX_WITH_CXX14_INTEGER_SEQUENCE
    SOURCE cmake/tests/cxx14_std_integer_sequence.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx14_std_is_final)
  add_hpx_config_test(HPX_WITH_CXX14_IS_FINAL
    SOURCE cmake/tests/cxx14_std_is_final.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx14_std_is_null_pointer)
  add_hpx_config_test(HPX_WITH_CXX14_IS_NULL_POINTER
    SOURCE cmake/tests/cxx14_std_is_null_pointer.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx14_std_result_of_sfinae)
  add_hpx_config_test(HPX_WITH_CXX14_RESULT_OF_SFINAE
    SOURCE cmake/tests/cxx14_std_result_of_sfinae.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx14_variable_templates)
  add_hpx_config_test(HPX_WITH_CXX14_VARIABLE_TEMPLATES
    SOURCE cmake/tests/cxx14_variable_templates.cpp
    FILE ${ARGN}
    CMAKECXXFEATURE cxx_variable_templates)
endmacro()

###############################################################################
macro(hpx_check_for_libfun_std_experimental_optional)
  add_hpx_config_test(HPX_WITH_LIBFUN_EXPERIMENTAL_OPTIONAL
    SOURCE cmake/tests/libfun_std_experimental_optional.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx17_fold_expressions)
  add_hpx_config_test(HPX_WITH_CXX17_FOLD_EXPRESSIONS
    SOURCE cmake/tests/cxx17_fold_expressions.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx17_fallthrough_attribute)
  add_hpx_config_test(HPX_WITH_CXX17_FALLTHROUGH_ATTRIBUTE
    SOURCE cmake/tests/cxx17_fallthrough_attribute.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_mm_prefetch)
  add_hpx_config_test(HPX_WITH_MM_PREFECTH
    SOURCE cmake/tests/mm_prefetch.cpp
    FILE ${ARGN})
endmacro()
