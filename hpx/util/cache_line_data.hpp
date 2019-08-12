//  Copyright (c) 2019 Hartmut Kaiser
//  Copyright (c) 2019 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_CACHE_LINE_DATA_AUG_12_2019_1043AM)
#define HPX_UTIL_CACHE_LINE_DATA_AUG_12_2019_1043AM

#include <hpx/config.hpp>
#include <hpx/concurrency/cache_aligned_data.hpp>

#include <cstddef>

namespace hpx { namespace util {
    ///////////////////////////////////////////////////////////////////////////
    // special struct to data type is cache line aligned and fully occupies a
    // cache line
#if defined(HPX_HAVE_CXX11_ALIGNAS) && defined(HPX_HAVE_CXX17_ALIGNED_NEW) &&  \
    !defined(__NVCC__)
    template <typename Data>
    struct alignas(threads::get_cache_line_size()) cache_line_data
    {
        Data data_;

        //  cppcheck-suppress unusedVariable
        char cacheline_pad[detail::get_cache_line_padding_size(sizeof(Data))];
    };
#else
    template <typename Data>
    struct cache_line_data
    {
        // pad to cache line size bytes
        Data data_;

        // cppcheck-suppress unusedVariable
        char cacheline_pad[detail::get_cache_line_padding_size(sizeof(Data))];
    };
#endif
}}    // namespace hpx::util

#endif
