//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_FUTURE_UNWRAP_GETTER_APR_29_2014_0955AM)
#define HPX_TRAITS_FUTURE_UNWRAP_GETTER_APR_29_2014_0955AM

#include <hpx/traits.hpp>
#include <hpx/config/forceinline.hpp>
#include <hpx/traits/future_traits.hpp>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Future, typename Enable>
    struct future_unwrap_getter
    {
        BOOST_FORCEINLINE
        typename future_traits<Future>::type operator()(Future f) const
        {
            return f.get();
        }
    };
}}

#endif

