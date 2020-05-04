//  Copyright (c) 2019-2020 Hartmut Kaiser
//  Copyright (c) 2019 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_CACHE_ALIGNED_DATA_MAR_17_2019_1103AM)
#define HPX_UTIL_CACHE_ALIGNED_DATA_MAR_17_2019_1103AM

#include <hpx/config.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx {

    namespace threads {
        ////////////////////////////////////////////////////////////////////////
        // abstract away cache-line size
        constexpr std::size_t get_cache_line_size()
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
                std::size_t data_size)
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

        // NOTE: We do not use alignas here because asking for overaligned
        // memory is significantly more expensive than asking for unaligned
        // memory. Padding the struct is cheaper and enough for internal
        // purposes.

        // NOTE: The implementations below are currently identical because of
        // the above issue. Both names are kept for compatibility.

        ///////////////////////////////////////////////////////////////////////////
        // special struct to ensure cache line alignment of a data type
        template <typename Data,
            typename NeedsPadding = typename detail::needs_padding<Data>::type>
        struct cache_aligned_data
        {
            // We have an explicit (non-default) constructor here to avoid for
            // the entire cache-line to be initialized by the compiler.
            cache_aligned_data()
              : data_()
            {
            }

            cache_aligned_data(Data&& data)
              : data_{std::move(data)}
            {
            }

            cache_aligned_data(Data const& data)
              : data_{data}
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
        struct cache_aligned_data<Data, std::false_type>
        {
            cache_aligned_data() = default;

            cache_aligned_data(Data&& data)
              : data_{std::move(data)}
            {
            }

            cache_aligned_data(Data const& data)
              : data_{data}
            {
            }

            // no need to pad to cache line size
            Data data_;
        };

        ///////////////////////////////////////////////////////////////////////////
        // special struct to ensure cache line alignment of a data type
        template <typename Data,
            typename NeedsPadding = typename detail::needs_padding<Data>::type>
        struct cache_aligned_data_derived : Data
        {
            // We have an explicit (non-default) constructor here to avoid for
            // the entire cache-line to be initialized by the compiler.
            cache_aligned_data_derived()
              : Data()
            {
            }

            cache_aligned_data_derived(Data&& data)
              : Data{std::move(data)}
            {
            }

            cache_aligned_data_derived(Data const& data)
              : Data{data}
            {
            }

            //  cppcheck-suppress unusedVariable
            char cacheline_pad[detail::get_cache_line_padding_size(
                // NOLINTNEXTLINE(bugprone-sizeof-expression)
                sizeof(Data))];
        };

        template <typename Data>
        struct cache_aligned_data_derived<Data, std::false_type> : Data
        {
            cache_aligned_data_derived() = default;

            cache_aligned_data_derived(Data&& data)
              : Data{std::move(data)}
            {
            }

            cache_aligned_data_derived(Data const& data)
              : Data{data}
            {
            }

            // no need to pad to cache line size
        };

        ///////////////////////////////////////////////////////////////////////////
        // special struct to data type is cache line aligned and fully occupies a
        // cache line
        template <typename Data>
        using cache_line_data = cache_aligned_data<Data>;
    }    // namespace util

}    // namespace hpx

#endif
