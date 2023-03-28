//  Copyright (c) 2019-2022 Hartmut Kaiser
//  Copyright (c) 2019 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/type_support/bit_cast.hpp>

#include <cstddef>
#include <new>
#include <type_traits>
#include <utility>

namespace hpx {

    namespace threads {

        ////////////////////////////////////////////////////////////////////////
        // abstract away cache-line size
        constexpr std::size_t get_cache_line_size() noexcept
        {
#if defined(HPX_HAVE_CXX17_HARDWARE_DESTRUCTIVE_INTERFERENCE_SIZE)
            return std::hardware_destructive_interference_size;
#else
#if defined(__s390__) || defined(__s390x__)
            return 256;    // assume 256 byte cache-line size
#elif defined(powerpc) || defined(__powerpc__) || defined(__ppc__)
            return 128;    // assume 128 byte cache-line size
#else
            return 64;    // assume 64 byte cache-line size
#endif
#endif
        }
    }    // namespace threads

    namespace util {

        namespace detail {

            // Computes the padding required to fill up a full cache line after
            // data_size bytes.
            constexpr std::size_t get_cache_line_padding_size(
                std::size_t data_size) noexcept
            {
                return (threads::get_cache_line_size() -
                           (data_size % threads::get_cache_line_size())) %
                    threads::get_cache_line_size();
            }

            template <typename Data>
            struct needs_padding
              : std::integral_constant<bool,
                    // NOLINTNEXTLINE(bugprone-sizeof-expression)
                    detail::get_cache_line_padding_size(sizeof(Data)) != 0>
            {
            };
        }    // namespace detail

        // Variable 'cacheline_pad' is uninitialized. Always initialize a member
        // variable
#if defined(HPX_MSVC)
#pragma warning(push)
#pragma warning(disable : 26495)
#endif

        // NOTE: We do not use alignas here because asking for overaligned
        // memory is significantly more expensive than asking for unaligned
        // memory. Padding the struct is cheaper and enough for internal
        // purposes.

        // NOTE: The implementations below are currently identical because of
        // the above issue. Both names are kept for compatibility.

        ///////////////////////////////////////////////////////////////////////////
        // special struct to ensure cache line alignment of a data type
        template <typename Data,
            bool NeedsPadding = detail::needs_padding<Data>::value>
        struct cache_aligned_data
        {
            // We have an explicit (default) constructor here to avoid for the
            // entire cache-line to be initialized by the compiler.

            constexpr cache_aligned_data() noexcept(    //-V730
                std::is_nothrow_default_constructible_v<Data>)
              : data_()
            {
            }

            template <typename... Ts,
                typename =
                    std::enable_if_t<std::is_constructible_v<Data, Ts&&...>>>
            cache_aligned_data(Ts&&... ts) noexcept(
                std::is_nothrow_constructible_v<Data, Ts&&...>)
              : data_(HPX_FORWARD(Ts, ts)...)
            {
            }

            // pad to cache line size bytes
            Data data_;

            //  cppcheck-suppress unusedVariable
            char cacheline_pad[detail::get_cache_line_padding_size(
                // NOLINTNEXTLINE(bugprone-sizeof-expression)
                sizeof(Data))];
        };

        template <typename Data>
        struct cache_aligned_data<Data, false>
        {
            constexpr cache_aligned_data() noexcept(
                std::is_nothrow_default_constructible_v<Data>)
              : data_()
            {
            }

            template <typename... Ts,
                typename =
                    std::enable_if_t<std::is_constructible_v<Data, Ts&&...>>>
            cache_aligned_data(Ts&&... ts) noexcept(
                std::is_nothrow_constructible_v<Data, Ts&&...>)
              : data_(HPX_FORWARD(Ts, ts)...)
            {
            }

            // no need to pad to cache line size
            Data data_;
        };

        ///////////////////////////////////////////////////////////////////////////
        // special struct to ensure cache line alignment of a data type
        template <typename Data,
            bool NeedsPadding = detail::needs_padding<Data>::value>
        struct cache_aligned_data_derived : Data
        {
            // We have an explicit (default) constructor here to avoid for the
            // entire cache-line to be initialized by the compiler.
            constexpr cache_aligned_data_derived() noexcept(    //-V730
                std::is_nothrow_default_constructible_v<Data>)
              : Data()
            {
            }

            template <typename... Ts,
                typename =
                    std::enable_if_t<std::is_constructible_v<Data, Ts&&...>>>
            cache_aligned_data_derived(Ts&&... ts) noexcept(
                std::is_nothrow_constructible_v<Data, Ts&&...>)
              : Data(HPX_FORWARD(Ts, ts)...)
            {
            }

            //  cppcheck-suppress unusedVariable
            char cacheline_pad[detail::get_cache_line_padding_size(
                // NOLINTNEXTLINE(bugprone-sizeof-expression)
                sizeof(Data))];
        };

        template <typename Data>
        struct cache_aligned_data_derived<Data, false> : Data
        {
            constexpr cache_aligned_data_derived() noexcept(
                std::is_nothrow_default_constructible_v<Data>)
              : Data()
            {
            }

            template <typename... Ts,
                typename =
                    std::enable_if_t<std::is_constructible_v<Data, Ts&&...>>>
            cache_aligned_data_derived(Ts&&... ts) noexcept(
                std::is_nothrow_constructible_v<Data, Ts&&...>)
              : Data(HPX_FORWARD(Ts, ts)...)
            {
            }

            // no need to pad to cache line size
        };

#if defined(HPX_MSVC)
#pragma warning(pop)
#endif

        ///////////////////////////////////////////////////////////////////////////
        // special struct to data type is cache line aligned and fully occupies a
        // cache line
        template <typename Data>
        using cache_line_data = cache_aligned_data<Data>;

        ///////////////////////////////////////////////////////////////////////////
        template <typename T>
        constexpr auto align_up(T value, std::size_t alignment) noexcept
        {
            return T(hpx::bit_cast<std::size_t>(value + (alignment - 1)) &
                ~(alignment - 1));
        }
    }    // namespace util

}    // namespace hpx
