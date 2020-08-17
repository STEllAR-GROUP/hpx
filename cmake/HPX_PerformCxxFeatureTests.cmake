# Copyright (c) 2007-2017 Hartmut Kaiser
# Copyright (c) 2011-2014 Thomas Heller
# Copyright (c) 2013-2016 Agustin Berge
# Copyright (c)      2017 Taeguk Kwon
# Copyright (c)      2020 Giannis Gonidelis
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# ##############################################################################
# C++ feature tests
# ##############################################################################
function(hpx_perform_cxx_feature_tests)
  hpx_check_for_cxx11_std_atomic(
    REQUIRED "HPX needs support for C++11 std::atomic"
  )

  # Separately check for 128 bit atomics
  hpx_check_for_cxx11_std_atomic_128bit(
    DEFINITIONS HPX_HAVE_CXX11_STD_ATOMIC_128BIT
  )

  hpx_check_for_cxx11_std_quick_exit(DEFINITIONS HPX_HAVE_CXX11_STD_QUICK_EXIT)

  hpx_check_for_cxx11_std_shared_ptr_lwg3018(
    DEFINITIONS HPX_HAVE_CXX11_STD_SHARED_PTR_LWG3018
  )

  hpx_check_for_c11_aligned_alloc(DEFINITIONS HPX_HAVE_C11_ALIGNED_ALLOC)

  hpx_check_for_cxx17_std_aligned_alloc(
    DEFINITIONS HPX_HAVE_CXX17_STD_ALIGNED_ALLOC
  )

  hpx_check_for_cxx17_filesystem(DEFINITIONS HPX_HAVE_CXX17_FILESYSTEM)

  hpx_check_for_cxx17_fold_expressions(
    DEFINITIONS HPX_HAVE_CXX17_FOLD_EXPRESSIONS
  )

  hpx_check_for_cxx17_fallthrough_attribute(
    DEFINITIONS HPX_HAVE_CXX17_FALLTHROUGH_ATTRIBUTE
  )

  hpx_check_for_cxx17_hardware_destructive_interference_size(
    DEFINITIONS HPX_HAVE_CXX17_HARDWARE_DESTRUCTIVE_INTERFERENCE_SIZE
  )

  hpx_check_for_cxx17_structured_bindings(
    DEFINITIONS HPX_HAVE_CXX17_STRUCTURED_BINDINGS
  )

  hpx_check_for_cxx17_if_constexpr(DEFINITIONS HPX_HAVE_CXX17_IF_CONSTEXPR)

  hpx_check_for_cxx17_inline_constexpr_variable(
    DEFINITIONS HPX_HAVE_CXX17_INLINE_CONSTEXPR_VARIABLE
  )

  hpx_check_for_cxx17_aligned_new(DEFINITIONS HPX_HAVE_CXX17_ALIGNED_NEW)

  hpx_check_for_cxx17_std_in_place_type_t(
    DEFINITIONS HPX_HAVE_CXX17_STD_IN_PLACE_TYPE_T
  )

  hpx_check_for_cxx17_std_variant(DEFINITIONS HPX_HAVE_CXX17_STD_VARIANT)

  hpx_check_for_cxx17_shared_ptr_array(
    DEFINITIONS HPX_HAVE_CXX17_SHARED_PTR_ARRAY
  )

  hpx_check_for_cxx17_maybe_unused(DEFINITIONS HPX_HAVE_CXX17_MAYBE_UNUSED)

  hpx_check_for_cxx17_deduction_guides(
    DEFINITIONS HPX_HAVE_CXX17_DEDUCTION_GUIDES
  )

  # we deliberately check for this functionality even for non-C++17
  # configurations as some compilers (notable gcc V7.x) require for noexcept
  # function specializations for actions even in C++11/14 mode
  hpx_check_for_cxx17_noexcept_functions_as_nontype_template_arguments(
    DEFINITIONS HPX_HAVE_CXX17_NOEXCEPT_FUNCTIONS_AS_NONTYPE_TEMPLATE_ARGUMENTS
  )

  hpx_check_for_cxx17_std_transform_scan(
    DEFINITIONS HPX_HAVE_CXX17_STD_TRANSFORM_SCAN_ALGORITHMS
  )

  hpx_check_for_cxx17_std_scan(DEFINITIONS HPX_HAVE_CXX17_STD_SCAN_ALGORITHMS)

  hpx_check_for_cxx17_std_nontype_template_parameter_auto(
    DEFINITIONS HPX_HAVE_CXX17_NONTYPE_TEMPLATE_PARAMETER_AUTO
  )

  # C++20 feature tests
  hpx_check_for_cxx20_coroutines(DEFINITIONS HPX_HAVE_CXX20_COROUTINES)

  hpx_check_for_cxx20_no_unique_address_attribute(
    DEFINITIONS HPX_HAVE_CXX20_NO_UNIQUE_ADDRESS_ATTRIBUTE
  )

  hpx_check_for_cxx20_std_disable_sized_sentinel_for(
    DEFINITIONS HPX_HAVE_CXX20_STD_DISABLE_SIZED_SENTINEL_FOR
  )

  hpx_check_for_cxx20_std_endian(DEFINITIONS HPX_HAVE_CXX20_STD_ENDIAN)

  # Check the availability of certain C++ builtins
  hpx_check_for_builtin_integer_pack(DEFINITIONS HPX_HAVE_BUILTIN_INTEGER_PACK)

  hpx_check_for_builtin_make_integer_seq(
    DEFINITIONS HPX_HAVE_BUILTIN_MAKE_INTEGER_SEQ
  )

  hpx_check_for_builtin_type_pack_element(
    DEFINITIONS HPX_HAVE_BUILTIN_TYPE_PACK_ELEMENT
  )

endfunction()
