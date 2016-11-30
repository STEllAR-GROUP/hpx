//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_VOID_GUARD_HPP
#define HPX_UTIL_VOID_GUARD_HPP

#include <hpx/config.hpp>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    // This utility simplifies templates returning compatible types
    //
    // Usage: return void_guard<Result>(), expr;
    // - Result != void -> return expr;
    // - Result == void -> return (void)expr;
    template <typename Result>
    struct void_guard
    {};

    template <>
    struct void_guard<void>
    {
        template <typename T>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        void operator,(T const&) const HPX_NOEXCEPT
        {}
    };
}}

#endif
