// Copyright (c) 2007-2012 Hartmut Kaiser
//
// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_BINARY_FILTER_FEB_14_2013_0809PM)
#define HPX_UTIL_BINARY_FILTER_FEB_14_2013_0809PM

namespace hpx { namespace util
{
    /////////////////////////////////////////////////////////////////////////
    struct binary_filter
    {
        virtual ~binary_filter() {}
        virtual std::size_t load(void* address, void const* src, std::size_t count) = 0;
        virtual std::size_t save(void* dest, void const* address, std::size_t count) = 0;
    };
}}

#endif
