/*=============================================================================
    Copyright (c) 2001-2011 Joel de Guzman
    Copyright (c) 2007-2021 Hartmut Kaiser

//  SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
==============================================================================*/
#pragma once

#include <hpx/config.hpp>

#if defined(HPX_MSVC)
#pragma warning(push)
#pragma warning(                                                               \
    disable : 4522)    // multiple assignment operators specified warning
#endif

namespace hpx::util {

    ////////////////////////////////////////////////////////////////////////////
    // We do not import fusion::unused_type anymore to avoid boost::fusion being
    // turned into an associate namespace, as this interferes with ADL in
    // unexpected ways. We rather copy the full unused_type implementation.
    ////////////////////////////////////////////////////////////////////////////
    struct unused_type
    {
        constexpr HPX_HOST_DEVICE HPX_FORCEINLINE unused_type() noexcept =
            default;

        constexpr HPX_HOST_DEVICE HPX_FORCEINLINE unused_type(
            unused_type const&) noexcept
        {
        }
        constexpr HPX_HOST_DEVICE HPX_FORCEINLINE unused_type(
            unused_type&&) noexcept
        {
        }

        template <typename T>
        /*implicit*/ constexpr HPX_HOST_DEVICE HPX_FORCEINLINE unused_type(
            T const&) noexcept
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
        constexpr HPX_HOST_DEVICE HPX_FORCEINLINE unused_type const& operator=(
            unused_type&&) const noexcept
        {
            return *this;
        }

        HPX_HOST_DEVICE HPX_FORCEINLINE unused_type& operator=(
            unused_type const&) noexcept
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
#else
    inline constexpr
#endif
    unused_type unused = unused_type();
}    // namespace hpx::util

//////////////////////////////////////////////////////////////////////////////
// use this to silence compiler warnings related to unused function arguments.
#if defined(__CUDA_ARCH__)
#define HPX_UNUSED(x) (void) x
#else
#define HPX_UNUSED(x) ::hpx::util::unused = (x)
#endif

#if defined(HPX_MSVC)
#pragma warning(pop)
#endif
