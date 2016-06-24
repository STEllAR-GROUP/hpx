//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_GET_AND_RESET_VALUE_FEB_27_2012_0248PM)
#define HPX_UTIL_GET_AND_RESET_VALUE_FEB_27_2012_0248PM

#include <boost/atomic.hpp>
#include <boost/cstdint.hpp>

namespace hpx { namespace util
{
    // helper function for counter evaluation
    inline boost::uint64_t get_and_reset_value(boost::uint64_t& value, bool reset)
    {
        boost::uint64_t result = value;
        if (reset)
            value = 0;
        return result;
    }

    inline boost::int64_t get_and_reset_value(boost::int64_t& value, bool reset)
    {
        boost::int64_t result = value;
        if (reset)
            value = 0;
        return result;
    }

    template <typename T>
    inline T get_and_reset_value(boost::atomic<T>& value, bool reset)
    {
        T result = value.load();
        if (reset)
            value.store(0);
        return result;
    }
}}

#endif
