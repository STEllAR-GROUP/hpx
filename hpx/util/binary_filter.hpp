// Copyright (c) 2007-2013 Hartmut Kaiser
//
// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_BINARY_FILTER_FEB_14_2013_0809PM)
#define HPX_UTIL_BINARY_FILTER_FEB_14_2013_0809PM

#include <memory>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    // Base class for all serialization filters.
    struct binary_filter
    {
        enum mode {
            favor_speed = 1,
            favor_memorysize = 2
        };

        virtual ~binary_filter() {}

        virtual std::size_t load(void* dst, std::size_t dst_count,
            void const* src, std::size_t src_count) = 0;
        virtual std::size_t save(void* dst, std::size_t dst_count,
            void const* src, std::size_t src_count) = 0;
        virtual std::size_t flush(void* dst, std::size_t dst_count) = 0;

        virtual void set_max_compression_length(std::size_t size) = 0;
        virtual void init_decompression_data(char const* buffer,
            std::size_t size, std::size_t decompressed_size) = 0;
    };
}}

#endif
