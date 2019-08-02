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

    ///////////////////////////////////////////////////////////////////////////
    // special struct to ensure cache line alignment of data
    // consisting of multiple elements (a tuple in effect)
    // The tuple is cache aligned and fully occupies 1 or more cache lines
    // use get<0>, get<1>, ... get<N> to access the internal elements
#if defined(HPX_HAVE_CXX11_ALIGNAS) && defined(HPX_HAVE_CXX17_ALIGNED_NEW) &&  \
    !defined(__NVCC__)
    template <typename ... Data>
    struct alignas(threads::get_cache_line_size()) cache_line_multidata
    {
#else
    template <typename ... Data>
    struct cache_line_multidata
    {
#endif
        template <typename ... T>
        cache_line_multidata(T...ts) : data_(ts...) {};

        template <std::size_t I>
        auto &get() {
            return std::get<I>(data_);
        }

        template <std::size_t I>
        auto const &get() const {
            return std::get<I>(data_);
        }

        // the storage of the data in tuple form
        std::tuple<Data...> data_;

        // pad to multiple of cache line size bytes
        // use zero padding if multiple of line size already
        static constexpr unsigned int padding1 = sizeof(std::tuple<Data...>);
        static constexpr unsigned int padding2 =
            (padding1==0) ? 0 : threads::get_cache_line_size() - padding1;
        char cacheline_pad[padding2];
    };
}}

#endif
