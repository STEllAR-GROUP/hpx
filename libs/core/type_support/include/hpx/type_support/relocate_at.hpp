//  Copyright (c) 2023 Isidoros Tsaousis-Seiras
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/type_support/construct_at.hpp>
#include <hpx/type_support/is_relocatable.hpp>
#include <hpx/type_support/is_trivially_relocatable.hpp>

#include <cstring>
#include <type_traits>

#if defined(HPX_HAVE_P1144_STD_RELOCATE_AT)
#include <memory>
#endif

namespace hpx {

    namespace detail {
        template <typename T>
        struct destroy_guard
        {
            T* t;
            explicit destroy_guard(T* t)
              : t(t)
            {
            }
            ~destroy_guard()
            {
                std::destroy_at(t);
            }
        };
    }    // namespace detail

#if defined(HPX_HAVE_P1144_STD_RELOCATE_AT)
    using std::relocate;
    using std::relocate_at;
#else

    namespace detail {

        /*
        The condition to use memmove is:
                hpx::is_trivially_relocatable_v<T> &&
                !std::is_volatile_v<T>

        The reason for the volatile check is that memmove is not allowed to
        change the value of a volatile object.
        */

        template <typename T>
        constexpr bool relocate_using_memmove =
            hpx::is_trivially_relocatable_v<T> && !std::is_volatile_v<T>;

        template <typename T,
            std::enable_if_t<relocate_using_memmove<T>, int> = 0>
        T* relocate_at_helper(T* src, T* dst) noexcept
        {
            std::memmove(dst, src, sizeof(T));
            return std::launder(dst);
        };

        // this is move and destroy
        template <typename T,
            std::enable_if_t<!relocate_using_memmove<T>, int> = 0>
        T* relocate_at_helper(T* src, T* dst) noexcept(
            // has non-throwing move constructor
            std::is_nothrow_move_constructible_v<T>)
        {
            destroy_guard g(src);
            return hpx::construct_at(dst, HPX_MOVE(*src));
        };
    }    // namespace detail

    template <typename T>
    T* relocate_at(T* src, T* dst) noexcept(
        // noexcept if the memmove path is taken or if the move path is noexcept
        noexcept(detail::relocate_at_helper(src, dst)))
    {
        static_assert(hpx::is_relocatable_v<T>,
            "new (dst) T(std::move(*src)) must be well-formed");

        return detail::relocate_at_helper(src, dst);
    }

    template <typename T>
    T relocate(T* src) noexcept(std::is_nothrow_move_constructible_v<T>)
    {
        static_assert(
            hpx::is_relocatable_v<T>, "T(std::move(*src)) must be well-formed");

        detail::destroy_guard g(src);
        return HPX_MOVE(*src);
    }

    /*
    Memmove codegen. This part relies on UB, so it's not used. It's here for
        reference. More info on this:

    https://quuxplusone.github.io/blog/2022/05/18/std-relocate/

    template <class T>
    T relocate(T* source)
    {
        auto magic = (T(*)(void*, size_t)) memcpy;
        return magic(source, sizeof(T));
    }
    */

#endif    // defined(HPX_HAVE_P1144_STD_RELOCATE_AT)

}    // namespace hpx
