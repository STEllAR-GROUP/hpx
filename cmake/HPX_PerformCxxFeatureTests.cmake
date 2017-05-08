# Copyright (c) 2007-2017 Hartmut Kaiser
# Copyright (c) 2011-2014 Thomas Heller
# Copyright (c) 2013-2016 Agustin Berge
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

################################################################################
# C++ feature tests
################################################################################
macro(hpx_perform_cxx_feature_tests)

  # Check the availability of certain C++11 language features
  hpx_check_for_cxx11_alias_templates(
    REQUIRED "HPX needs support for C++11 alias templates")

  hpx_check_for_cxx11_auto(
    REQUIRED "HPX needs support for C++11 auto")

  hpx_check_for_cxx11_constexpr(
    DEFINITIONS HPX_HAVE_CXX11_CONSTEXPR)

  hpx_check_for_cxx11_decltype(
    REQUIRED "HPX needs support for C++11 decltype")
  hpx_add_config_define(BOOST_RESULT_OF_USE_DECLTYPE)

  hpx_check_for_cxx11_decltype_n3276(
    DEFINITIONS HPX_HAVE_CXX11_DECLTYPE_N3276)

  hpx_check_for_cxx11_sfinae_expression(
    DEFINITIONS HPX_HAVE_CXX11_SFINAE_EXPRESSION)

  hpx_check_for_cxx11_defaulted_functions(
    DEFINITIONS HPX_HAVE_CXX11_DEFAULTED_FUNCTIONS)

  hpx_check_for_cxx11_deleted_functions(
    REQUIRED "HPX needs support for C++11 deleted functions")

  hpx_check_for_cxx11_explicit_cvt_ops(
    REQUIRED "HPX needs support for C++11 explicit conversion operators")

  hpx_check_for_cxx11_explicit_variadic_templates(
    DEFINITIONS HPX_HAVE_CXX11_EXPLICIT_VARIADIC_TEMPLATES)

  hpx_check_for_cxx11_extended_friend_declarations(
    DEFINITIONS HPX_HAVE_CXX11_EXTENDED_FRIEND_DECLARATIONS)

  hpx_check_for_cxx11_function_template_default_args(
    REQUIRED "HPX needs support for C++11 defaulted function template arguments")

  hpx_check_for_cxx11_inline_namespaces(
    DEFINITIONS HPX_HAVE_CXX11_INLINE_NAMESPACES)

  hpx_check_for_cxx11_lambdas(
    DEFINITIONS HPX_HAVE_CXX11_LAMBDAS)

  hpx_check_for_cxx11_noexcept(
    DEFINITIONS HPX_HAVE_CXX11_NOEXCEPT)

  hpx_check_for_cxx11_nullptr(
    REQUIRED "HPX needs support for C++11 nullptr")

  hpx_check_for_cxx11_nsdmi(
    DEFINITIONS HPX_HAVE_CXX11_NSDMI)

  hpx_check_for_cxx11_range_based_for(
    REQUIRED "HPX needs support for C++11 range-based for-loop")

  hpx_check_for_cxx11_rvalue_references(
    REQUIRED "HPX needs support for C++11 rvalue references")

  hpx_check_for_cxx11_scoped_enums(
    REQUIRED "HPX needs support for C++11 scoped enums")

  hpx_check_for_cxx11_static_assert(
    REQUIRED "HPX needs support for C++11 static_assert")

  hpx_check_for_cxx11_variadic_macros(
    REQUIRED "HPX needs support for C++11 variadic macros")

  hpx_check_for_cxx11_variadic_templates(
    REQUIRED "HPX needs support for C++11 variadic templates")

  # Check the availability of certain C++11 library features
  hpx_check_for_cxx11_std_array(
    DEFINITIONS HPX_HAVE_CXX11_STD_ARRAY)

  hpx_check_for_cxx11_std_chrono(
    REQUIRED "HPX needs support for C++11 std::chrono")

  hpx_check_for_cxx11_std_cstdint(
    REQUIRED "HPX needs support for C++11 std::[u]intX_t")

  hpx_check_for_cxx11_std_initializer_list(
    REQUIRED "HPX needs support for C++11 std::initializer_list")

  hpx_check_for_cxx11_std_is_bind_expression(
    DEFINITIONS HPX_HAVE_CXX11_STD_IS_BIND_EXPRESSION)

  hpx_check_for_cxx11_std_is_placeholder(
    DEFINITIONS HPX_HAVE_CXX11_STD_IS_PLACEHOLDER)

  hpx_check_for_cxx11_std_is_trivially_copyable(
    DEFINITIONS HPX_HAVE_CXX11_STD_IS_TRIVIALLY_COPYABLE)

  hpx_check_for_cxx11_std_lock_guard(
    REQUIRED "HPX needs support for C++11 std::lock_guard")

  hpx_check_for_cxx11_std_reference_wrapper(
    REQUIRED "HPX needs support for C++11 std::ref and std::reference_wrapper")

  hpx_check_for_cxx11_std_shared_ptr(
    REQUIRED "HPX needs support for C++11 std::shared_ptr")

  hpx_check_for_cxx11_std_thread(
    DEFINITIONS HPX_HAVE_CXX11_STD_THREAD)

  hpx_check_for_cxx11_std_to_string(
    REQUIRED "HPX needs support for C++11 std::to_string")

  hpx_check_for_cxx11_std_unique_lock(
    REQUIRED "HPX needs support for C++11 std::unique_lock")

  hpx_check_for_cxx11_std_unique_ptr(
    REQUIRED "HPX needs support for C++11 std::unique_ptr")

  hpx_check_for_cxx11_std_unordered_map(
    REQUIRED "HPX needs support for C++11 std::unordered_map")

  hpx_check_for_cxx11_std_unordered_set(
    REQUIRED "HPX needs support for C++11 std::unordered_set")

  hpx_check_for_cxx11_std_type_traits(
    DEFINITIONS HPX_HAVE_CXX11_STD_TYPE_TRAITS)

  if(HPX_WITH_CXX1Y OR HPX_WITH_CXX14)
    # Check the availability of certain C++14 language features
    hpx_check_for_cxx14_constexpr(
      DEFINITIONS HPX_HAVE_CXX14_CONSTEXPR)

    hpx_check_for_cxx14_lambdas(
      DEFINITIONS HPX_HAVE_CXX14_LAMBDAS)

    # Check the availability of certain C++14 library features
    hpx_check_for_cxx14_std_integer_sequence(
      DEFINITIONS HPX_HAVE_CXX14_STD_INTEGER_SEQUENCE)

    hpx_check_for_cxx14_std_is_final(
      DEFINITIONS HPX_HAVE_CXX14_STD_IS_FINAL)

    hpx_check_for_cxx14_std_is_null_pointer(
      DEFINITIONS HPX_HAVE_CXX14_STD_IS_NULL_POINTER)

    hpx_check_for_cxx14_std_result_of_sfinae(
      DEFINITIONS HPX_HAVE_CXX14_STD_RESULT_OF_SFINAE)
  endif()

  # check for experimental facilities
  if(HPX_WITH_CXX1Y OR HPX_WITH_CXX14)
    # check for Library Fundamentals TS v2's experimental/optional only if in
    # C++1y or C++14 mode.
    hpx_check_for_libfun_std_experimental_optional(
      DEFINITIONS HPX_HAVE_LIBFUN_STD_EXPERIMENTAL_OPTIONAL)
  endif()
endmacro()

