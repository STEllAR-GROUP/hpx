# Copyright (c) 2007-2022 Hartmut Kaiser
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

  set(atomics_additional_flags COMPILE_DEFINITIONS
                               HPX_HAVE_CXX11_ATOMIC_FLAG_INIT
  )
  if(HPX_WITH_CXX_STANDARD GREATER_EQUAL 20)
    # ATOMIC_FLAG_INIT is deprecated starting C++20. Here we check whether using
    # it will cause failures (-Werror,-Wdeprecated-pragma), so we can disable
    # its use in the test for C++11 atomics below.
    hpx_check_for_cxx11_atomic_init_flag(
      DEFINITIONS HPX_HAVE_CXX11_ATOMIC_FLAG_INIT
    )
    if(NOT HPX_WITH_CXX11_ATOMIC_FLAG_INIT)
      set(atomics_additional_flags)
    endif()
  else()
    # for anything between C++11 and C++17, ATOMIC_FLAG_INIT should be used.
    hpx_add_config_define(HPX_HAVE_CXX11_ATOMIC_FLAG_INIT)
  endif()

  hpx_check_for_cxx11_std_atomic(
    REQUIRED "HPX needs support for C++11 std::atomic"
    ${atomics_additional_flags}
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

  hpx_check_for_cxx17_std_execution_policies(
    DEFINITIONS HPX_HAVE_CXX17_STD_EXECUTION_POLICES
  )

  hpx_check_for_cxx17_filesystem(DEFINITIONS HPX_HAVE_CXX17_FILESYSTEM)

  hpx_check_for_cxx17_hardware_destructive_interference_size(
    DEFINITIONS HPX_HAVE_CXX17_HARDWARE_DESTRUCTIVE_INTERFERENCE_SIZE
  )

  hpx_check_for_cxx17_aligned_new(DEFINITIONS HPX_HAVE_CXX17_ALIGNED_NEW)

  hpx_check_for_cxx17_shared_ptr_array(
    DEFINITIONS HPX_HAVE_CXX17_SHARED_PTR_ARRAY
  )

  hpx_check_for_cxx17_std_transform_scan(
    DEFINITIONS HPX_HAVE_CXX17_STD_TRANSFORM_SCAN_ALGORITHMS
  )

  hpx_check_for_cxx17_std_scan(DEFINITIONS HPX_HAVE_CXX17_STD_SCAN_ALGORITHMS)

  hpx_check_for_cxx17_copy_elision(DEFINITIONS HPX_HAVE_CXX17_COPY_ELISION)

  hpx_check_for_cxx17_optional_copy_elision(
    DEFINITIONS HPX_HAVE_CXX17_OPTIONAL_COPY_ELISION
  )

  # C++20 feature tests
  if(MSVC_VERSION GREATER_EQUAL 1929)
    # MSVC supports this attribute for all versions starting VS2019 v16.10 see
    # https://devblogs.microsoft.com/cppblog/msvc-cpp20-and-the-std-cpp20-switch/
    hpx_check_for_cxx20_no_unique_address_attribute(
      DEFINITIONS HPX_HAVE_MSVC_NO_UNIQUE_ADDRESS_ATTRIBUTE
    )
  endif()

  if(HPX_WITH_CXX_STANDARD GREATER_EQUAL 20)
    hpx_check_for_cxx20_coroutines(DEFINITIONS HPX_HAVE_CXX20_COROUTINES)

    hpx_check_for_cxx20_experimental_simd(
      DEFINITIONS HPX_HAVE_CXX20_EXPERIMENTAL_SIMD
    )

    hpx_check_for_cxx20_lambda_capture(
      DEFINITIONS HPX_HAVE_CXX20_LAMBDA_CAPTURE
    )

    hpx_check_for_cxx20_source_location(
      DEFINITIONS HPX_HAVE_CXX20_SOURCE_LOCATION
    )

    hpx_check_for_cxx20_perfect_pack_capture(
      DEFINITIONS HPX_HAVE_CXX20_PERFECT_PACK_CAPTURE
    )

    if(NOT MSVC) # see above
      hpx_check_for_cxx20_no_unique_address_attribute(
        DEFINITIONS HPX_HAVE_CXX20_NO_UNIQUE_ADDRESS_ATTRIBUTE
      )
    endif()

    hpx_check_for_cxx20_paren_initialization_of_aggregates(
      DEFINITIONS HPX_HAVE_CXX20_PAREN_INITIALIZATION_OF_AGGREGATES
    )

    hpx_check_for_cxx20_std_disable_sized_sentinel_for(
      DEFINITIONS HPX_HAVE_CXX20_STD_DISABLE_SIZED_SENTINEL_FOR
    )

    hpx_check_for_cxx20_std_endian(DEFINITIONS HPX_HAVE_CXX20_STD_ENDIAN)

    hpx_check_for_cxx20_std_execution_policies(
      DEFINITIONS HPX_HAVE_CXX20_STD_EXECUTION_POLICES
    )

    hpx_check_for_cxx20_std_ranges_iter_swap(
      DEFINITIONS HPX_HAVE_CXX20_STD_RANGES_ITER_SWAP
    )

    hpx_check_for_cxx20_trivial_virtual_destructor(
      DEFINITIONS HPX_HAVE_CXX20_TRIVIAL_VIRTUAL_DESTRUCTOR
    )

    hpx_check_for_cxx20_std_construct_at(
      DEFINITIONS HPX_HAVE_CXX20_STD_CONSTRUCT_AT
    )

    hpx_check_for_cxx20_std_default_sentinel(
      DEFINITIONS HPX_HAVE_CXX20_STD_DEFAULT_SENTINEL
    )

    hpx_check_for_cxx20_std_bit_cast(DEFINITIONS HPX_HAVE_CXX20_STD_BIT_CAST)
  endif()

  if(HPX_WITH_CXX_STANDARD GREATER_EQUAL 23)
    hpx_check_for_cxx23_std_generator(DEFINITIONS HPX_HAVE_CXX23_STD_GENERATOR)
  endif()

  hpx_check_for_cxx_lambda_capture_decltype(
    DEFINITIONS HPX_HAVE_CXX_LAMBDA_CAPTURE_DECLTYPE
  )

  # Check the availability of certain C++ builtins
  hpx_check_for_builtin_integer_pack(DEFINITIONS HPX_HAVE_BUILTIN_INTEGER_PACK)

  hpx_check_for_builtin_make_integer_seq(
    DEFINITIONS HPX_HAVE_BUILTIN_MAKE_INTEGER_SEQ
  )

  if(HPX_WITH_CUDA)
    hpx_check_for_builtin_make_integer_seq_cuda(
      DEFINITIONS HPX_HAVE_BUILTIN_MAKE_INTEGER_SEQ_CUDA
    )
  endif()

  hpx_check_for_builtin_type_pack_element(
    DEFINITIONS HPX_HAVE_BUILTIN_TYPE_PACK_ELEMENT
  )

  if(HPX_WITH_CUDA)
    hpx_check_for_builtin_type_pack_element_cuda(
      DEFINITIONS HPX_HAVE_BUILTIN_TYPE_PACK_ELEMENT_CUDA
    )
  endif()

  hpx_check_for_builtin_forward_move(DEFINITIONS HPX_HAVE_BUILTIN_FORWARD_MOVE)

endfunction()
