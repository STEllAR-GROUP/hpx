//  Copyright (c) 2019 Hartmut Kaiser
//  Copyright (c) 2019 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_CACHE_ALIGNED_DATA_MAR_17_2019_1103AM)
#define HPX_UTIL_CACHE_ALIGNED_DATA_MAR_17_2019_1103AM

#include <hpx/config.hpp>
#include <hpx/runtime/threads/topology.hpp>

namespace hpx { namespace util
{
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
        char cacheline_pad[threads::get_cache_line_size() -
            (sizeof(Data) % threads::get_cache_line_size())];
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
        char cacheline_pad[threads::get_cache_line_size() -
            (sizeof(Data) % threads::get_cache_line_size())];

    };
#else
    template <typename Data>
    struct cache_line_data
    {
        // pad to cache line size bytes
        Data data_;
        char cacheline_pad[threads::get_cache_line_size() -
            (sizeof(Data) % threads::get_cache_line_size())];
    };
#endif
}}

#endif
