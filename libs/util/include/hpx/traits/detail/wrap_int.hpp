//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_WRAP_INT_JUN_03_2016_0835PM)
#define HPX_TRAITS_WRAP_INT_JUN_03_2016_0835PM

#include <hpx/config.hpp>

namespace hpx { namespace traits { namespace detail
{
    // wraps int so that int argument is favored over wrap_int
    struct wrap_int
    {
        HPX_CONSTEXPR wrap_int(int) {}
    };
}}}

#endif
