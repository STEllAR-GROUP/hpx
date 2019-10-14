//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_CACHE_POLICIES_ALWAYS_NOV_19_2008_0803AM)
#define HPX_UTIL_CACHE_POLICIES_ALWAYS_NOV_19_2008_0803AM

#include <hpx/config.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util { namespace cache { namespace policies {
    ///////////////////////////////////////////////////////////////////////////
    template <typename Entry>
    struct always
    {
        bool operator()(Entry const&)
        {
            return true;    // always true
        }
    };
}}}}    // namespace hpx::util::cache::policies

#endif
