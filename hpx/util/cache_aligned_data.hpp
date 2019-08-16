//  Copyright (c) 2019 Hartmut Kaiser
//  Copyright (c) 2019 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_CACHE_ALIGNED_DATA_MAR_17_2019_1103AM)
#define HPX_UTIL_CACHE_ALIGNED_DATA_MAR_17_2019_1103AM

#include <hpx/config.hpp>
#include <hpx/runtime/threads/topology.hpp>

#include <cstddef>

namespace hpx { namespace util {
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
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    // special struct to ensure cache line alignment of a data type
#if defined(HPX_HAVE_CXX11_ALIGNAS) && defined(HPX_HAVE_CXX17_ALIGNED_NEW) &&  \
    !defined(__NVCC__)
    template <typename Data>
    struct alignas(threads::get_cache_line_size()) cache_aligned_data
    {
        Data data_;
    };
#else
    template <typename Data>
    struct cache_aligned_data
    {
        // pad to cache line size bytes
        Data data_;
        char cacheline_pad[detail::get_cache_line_padding_size(sizeof(Data))];
    };
#endif

    ///////////////////////////////////////////////////////////////////////////
    // special struct to data type is cache line aligned and fully occupies a
    // cache line
#if defined(HPX_HAVE_CXX11_ALIGNAS) && defined(HPX_HAVE_CXX17_ALIGNED_NEW) &&  \
    !defined(__NVCC__)
    template <typename Data>
    struct alignas(threads::get_cache_line_size()) cache_line_data
    {
        Data data_;
        char cacheline_pad[detail::get_cache_line_padding_size(sizeof(Data))];
    };
#else
    template <typename Data>
    struct cache_line_data
    {
        // pad to cache line size bytes
        Data data_;
        char cacheline_pad[detail::get_cache_line_padding_size(sizeof(Data))];
    };
#endif
}}    // namespace hpx::util

#endif
