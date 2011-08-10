//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THROTTLE_STUBS_AUG_09_2011_0703PM)
#define HPX_THROTTLE_STUBS_AUG_09_2011_0703PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future_value.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>

#include "../server/throttle.hpp"

namespace throttle { namespace stubs
{
    ///////////////////////////////////////////////////////////////////////////
    struct throttle : hpx::components::stubs::stub_base<server::throttle>
    {
    };
}}

#endif
