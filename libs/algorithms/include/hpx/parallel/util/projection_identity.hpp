//  Copyright (c) 2015-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_UTIL_PROJECTION_IDENTITY_JUL_18_2015_1105AM)
#define HPX_PARALLEL_UTIL_PROJECTION_IDENTITY_JUL_18_2015_1105AM

#include <hpx/config.hpp>

#include <utility>

namespace hpx { namespace parallel { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    struct projection_identity
    {
        template <typename T>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        T && operator()(T && val) const
        {
            return std::forward<T>(val);
        }
    };
}}}

#endif
