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

    ///////////////////////////////////////////////////////////////////////////
    // special struct to ensure cache line alignment of data
    // consisting of multiple elements (a tuple in effect)
    // The tuple is cache aligned and fully occupies 1 or more cache lines
    // use get<0>, get<1>, ... get<N> to access the internal elements
#if defined(HPX_HAVE_CXX11_ALIGNAS) && defined(HPX_HAVE_CXX17_ALIGNED_NEW) &&  \
    defined(HPX_HAVE_CXX17_FOLD_EXPRESSIONS) && !defined(__NVCC__)
    template <typename ... Data>
    struct alignas(threads::get_cache_line_size()) cache_line_multidata
    {
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

        // pad to cache line size bytes
        // todo remove wasted line if size is multiple of line size
        std::tuple<Data...> data_;
        char cacheline_pad[threads::get_cache_line_size() - 
            ((0 + ... + sizeof(Data)) % threads::get_cache_line_size())];
    };

#elif defined(HPX_HAVE_CXX14_STD_INTEGER_SEQUENCE)
    namespace detail
    {
        // helpers to get tuple pack size

        template<typename... Args>
        constexpr size_t add(Args... args);

        template<typename T, typename... Args>
        constexpr size_t add(T t, Args... args)
        {
          return t + add(args...);
        }
        template<>
        constexpr size_t add() {
          return 0;
        }

        template <typename... Args>
        static constexpr size_t PackSizeInBytes()
        {
            return add(sizeof(Args)...);
        }
    }

    template <typename ... Data>
    struct cache_line_multidata
    {
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

        // pad to multiple of cache line size bytes
        std::tuple<Data...> data_;
        char cacheline_pad[threads::get_cache_line_size() -
            (detail::PackSizeInBytes<Data...>() % threads::get_cache_line_size())];
    };
#else
    // not supported pre C++14
#endif

}}

#endif
