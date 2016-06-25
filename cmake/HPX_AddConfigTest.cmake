# Copyright (c) 2014 Thomas Heller
# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_ADDCONFIGTEST_LOADED TRUE)

macro(add_hpx_config_test variable)
  set(options FILE EXECUTE)
  set(one_value_args SOURCE ROOT)
  set(multi_value_args INCLUDE_DIRECTORIES LINK_DIRECTORIES COMPILE_DEFINITIONS LIBRARIES ARGS DEFINITIONS REQUIRED)
  cmake_parse_arguments(${variable} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(NOT DEFINED ${variable})
    file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/config_tests")

    string(TOUPPER "${variable}" variable_lc)
    if(${variable}_FILE)
      if(${variable}_ROOT)
        set(test_source "${${variable}_ROOT}/share/hpx-${HPX_VERSION}/${${variable}_SOURCE}")
      else()
        set(test_source "${PROJECT_SOURCE_DIR}/${${variable}_SOURCE}")
      endif()
    else()
      set(test_source
          "${PROJECT_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/config_tests/${variable_lc}.cpp")
      file(WRITE "${test_source}"
           "${${variable}_SOURCE}\n")
    endif()
    set(test_binary ${PROJECT_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/config_tests/${variable_lc})

    get_directory_property(CONFIG_TEST_INCLUDE_DIRS INCLUDE_DIRECTORIES)
    get_directory_property(CONFIG_TEST_LINK_DIRS LINK_DIRECTORIES)
    set(COMPILE_DEFINITIONS_TMP)
    set(CONFIG_TEST_COMPILE_DEFINITIONS)
    get_directory_property(COMPILE_DEFINITIONS_TMP COMPILE_DEFINITIONS)
    foreach(def ${COMPILE_DEFINITIONS_TMP})
      set(CONFIG_TEST_COMPILE_DEFINITIONS "${CONFIG_TEST_COMPILE_DEFINITIONS} -D${def}")
    endforeach()
    get_property(HPX_TARGET_COMPILE_OPTIONS_VAR GLOBAL PROPERTY HPX_TARGET_COMPILE_OPTIONS)
    foreach(_flag ${HPX_TARGET_COMPILE_OPTIONS_VAR})
      if(NOT "${_flag}" MATCHES "^\\$.*")
        set(CONFIG_TEST_COMPILE_DEFINITIONS "${CONFIG_TEST_COMPILE_DEFINITIONS} ${_flag}")
      endif()
    endforeach()

    set(CONFIG_TEST_INCLUDE_DIRS ${CONFIG_TEST_INCLUDE_DIRS} ${${variable}_INCLUDE_DIRS})
    set(CONFIG_TEST_LINK_DIRS ${CONFIG_TEST_LINK_DIRS} ${${variable}_LINK_DIRS})

    set(CONFIG_TEST_COMPILE_DEFINITIONS ${CONFIG_TEST_COMPILE_DEFINITIONS} ${${variable}_COMPILE_DEFINITIONS})
    set(CONFIG_TEST_LINK_LIBRARIES ${HPX_LIBRARIES} ${${variable}_LIBRARIES})

    if(${variable}_EXECUTE)
      if(NOT CMAKE_CROSSCOMPILING)
        try_run(${variable}_RUN_RESULT ${variable}_COMPILE_RESULT
          ${PROJECT_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/config_tests
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
        ${PROJECT_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/config_tests
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
    set(_run_msg "pre-set to ${${variable}}")
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
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_auto)
  add_hpx_config_test(HPX_WITH_CXX11_AUTO
    SOURCE cmake/tests/cxx11_auto.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_constexpr)
  add_hpx_config_test(HPX_WITH_CXX11_CONSTEXPR
    SOURCE cmake/tests/cxx11_constexpr.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_decltype)
  add_hpx_config_test(HPX_WITH_CXX11_DECLTYPE
    SOURCE cmake/tests/cxx11_decltype.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_decltype_n3276)
  add_hpx_config_test(HPX_WITH_CXX11_DECLTYPE_N3276
    SOURCE cmake/tests/cxx11_decltype_n3276.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_sfinae_expression)
  add_hpx_config_test(HPX_WITH_CXX11_SFINAE_EXPRESSION
    SOURCE cmake/tests/cxx11_sfinae_expression.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_defaulted_functions)
  add_hpx_config_test(HPX_WITH_CXX11_DEFAULTED_FUNCTIONS
    SOURCE cmake/tests/cxx11_defaulted_functions.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_deleted_functions)
  add_hpx_config_test(HPX_WITH_CXX11_DELETED_FUNCTIONS
    SOURCE cmake/tests/cxx11_deleted_functions.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_explicit_cvt_ops)
  add_hpx_config_test(HPX_WITH_CXX11_EXPLICIT_CONVERSION_OPERATORS
    SOURCE cmake/tests/cxx11_explicit_cvt_ops.cpp
    FILE ${ARGN})
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
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_function_template_default_args)
  add_hpx_config_test(HPX_WITH_CXX11_FUNCTION_TEMPLATE_DEFAULT_ARGS
    SOURCE cmake/tests/cxx11_function_template_default_args.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_initializer_list)
  add_hpx_config_test(HPX_WITH_CXX11_INITIALIZER_LIST
    SOURCE cmake/tests/cxx11_initializer_list.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_inline_namespaces)
  add_hpx_config_test(HPX_WITH_CXX11_INLINE_NAMESPACES
    SOURCE cmake/tests/cxx11_inline_namespaces.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_lambdas)
  add_hpx_config_test(HPX_WITH_CXX11_LAMBDAS
    SOURCE cmake/tests/cxx11_lambdas.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_noexcept)
  add_hpx_config_test(HPX_WITH_CXX11_NOEXCEPT
    SOURCE cmake/tests/cxx11_noexcept.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_nullptr)
  add_hpx_config_test(HPX_WITH_CXX11_NULLPTR
    SOURCE cmake/tests/cxx11_nullptr.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_range_based_for)
  add_hpx_config_test(HPX_WITH_CXX11_RANGE_BASED_FOR
    SOURCE cmake/tests/cxx11_range_based_for.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_rvalue_references)
  add_hpx_config_test(HPX_WITH_CXX11_RVALUE_REFERENCES
    SOURCE cmake/tests/cxx11_rvalue_references.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_scoped_enums)
  add_hpx_config_test(HPX_WITH_CXX11_SCOPED_ENUMS
    SOURCE cmake/tests/cxx11_scoped_enums.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_static_assert)
  add_hpx_config_test(HPX_WITH_CXX11_STATIC_ASSERT
    SOURCE cmake/tests/cxx11_static_assert.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_variadic_macros)
  add_hpx_config_test(HPX_WITH_CXX11_VARIADIC_MACROS
    SOURCE cmake/tests/cxx11_variadic_macros.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_variadic_templates)
  add_hpx_config_test(HPX_WITH_CXX11_VARIADIC_TEMPLATES
    SOURCE cmake/tests/cxx11_variadic_templates.cpp
    FILE ${ARGN})
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
macro(hpx_check_for_cxx11_std_to_string)
  add_hpx_config_test(HPX_WITH_CXX11_TO_STRING
    SOURCE cmake/tests/cxx11_std_to_string.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_cxx11_std_type_traits)
  add_hpx_config_test(HPX_WITH_CXX11_TYPE_TRAITS
    SOURCE cmake/tests/cxx11_std_type_traits.cpp
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
macro(hpx_check_for_cxx_experimental_std_optional)
  add_hpx_config_test(HPX_WITH_CXX1Y_EXPERIMENTAL_OPTIONAL
    SOURCE cmake/tests/cxx1y_experimental_std_optional.cpp
    FILE ${ARGN})
endmacro()

###############################################################################
macro(hpx_check_for_mm_prefetch)
  add_hpx_config_test(HPX_WITH_MM_PREFECTH
    SOURCE cmake/tests/mm_prefetch.cpp
    FILE ${ARGN})
endmacro()
