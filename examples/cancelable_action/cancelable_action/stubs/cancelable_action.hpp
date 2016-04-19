//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_EXAMPLE_CANCELABLE_ACTION_STUB_APR_21_2012_1055AM)
#define HPX_EXAMPLE_CANCELABLE_ACTION_STUB_APR_21_2012_1055AM

#include <hpx/hpx.hpp>

#include "../server/cancelable_action.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace examples { namespace stubs
{
    ///////////////////////////////////////////////////////////////////////////
    struct cancelable_action
      : hpx::components::stub_base<server::cancelable_action>
    {
        // Do some lengthy work
        static hpx::lcos::future<void>
        do_it_async(hpx::naming::id_type const& gid)
        {
            typedef server::cancelable_action::do_it_action action_type;
            return hpx::async<action_type>(gid);
        }

        static void do_it(hpx::naming::id_type const& gid,
            hpx::error_code& ec = hpx::throws)
        {
            do_it_async(gid).get(ec);
        }

        // Cancel the lengthy action above
        static void cancel_it(hpx::naming::id_type const& gid)
        {
            typedef server::cancelable_action::cancel_it_action action_type;
            hpx::apply<action_type>(gid);
        }
    };
}}

#endif

