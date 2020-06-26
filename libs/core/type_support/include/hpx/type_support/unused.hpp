/*=============================================================================
    Copyright (c) 2001-2011 Joel de Guzman
    Copyright (c) 2007-2019 Hartmut Kaiser

//  SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
==============================================================================*/
#pragma once

// clang-format off
#include <hpx/config.hpp>
#if defined(HPX_MSVC)
# pragma warning(push)
# pragma warning(disable: 4522) // multiple assignment operators specified warning
#endif
// clang-format on

namespace hpx { namespace util {
    ///////////////////////////////////////////////////////////////////////////
    // We do not import fusion::unused_type anymore to avoid boost::fusion
    // being turned into an associate namespace, as this interferes with ADL
    // in unexpected ways. We rather copy the full unused_type implementation.
    ///////////////////////////////////////////////////////////////////////////
    struct unused_type
    {
        constexpr HPX_HOST_DEVICE HPX_FORCEINLINE unused_type() noexcept {}

        constexpr HPX_HOST_DEVICE HPX_FORCEINLINE unused_type(
            unused_type const&)
        {
        }
        constexpr HPX_HOST_DEVICE HPX_FORCEINLINE unused_type(unused_type&&) {}

        template <typename T>
        constexpr HPX_HOST_DEVICE HPX_FORCEINLINE unused_type(T const&) noexcept
        {
        }

        template <typename T>
        constexpr HPX_HOST_DEVICE HPX_FORCEINLINE unused_type const& operator=(
            T const&) const noexcept
        {
            return *this;
        }

        template <typename T>
        HPX_HOST_DEVICE HPX_FORCEINLINE unused_type& operator=(
            T const&) noexcept
        {
            return *this;
        }

        constexpr HPX_HOST_DEVICE HPX_FORCEINLINE unused_type const& operator=(
            unused_type const&) const noexcept
        {
            return *this;
        }

        HPX_HOST_DEVICE HPX_FORCEINLINE unused_type& operator=(
            unused_type const&) noexcept
        {
            return *this;
        }

        constexpr HPX_HOST_DEVICE HPX_FORCEINLINE unused_type const& operator=(
            unused_type&&) const noexcept
        {
            return *this;
        }

        HPX_HOST_DEVICE HPX_FORCEINLINE unused_type& operator=(
            unused_type&&) noexcept
        {
            return *this;
        }
    };

#if defined(HPX_MSVC_NVCC)
    HPX_CONSTANT
#endif
    constexpr unused_type unused = unused_type();
}}    // namespace hpx::util

//////////////////////////////////////////////////////////////////////////////
// use this to silence compiler warnings related to unused function arguments.
#if defined(__CUDA_ARCH__)
#define HPX_UNUSED(x) (void) x
#else
#define HPX_UNUSED(x) ::hpx::util::unused = (x)
#endif

// clang-format off
/////////////////////////////////////////////////////////////
// use this to silence compiler warnings for global variables
#ifdef HPX_HAVE_CXX17_MAYBE_UNUSED
#  define HPX_MAYBE_UNUSED [[maybe_unused]]
#else
#  if defined(HPX_GCC_VERSION)
#    define HPX_MAYBE_UNUSED __attribute__((unused))
#  else
#    define HPX_MAYBE_UNUSED /* empty */
#  endif
#endif
// clang-format on

#if defined(HPX_MSVC)
#pragma warning(pop)
#endif
