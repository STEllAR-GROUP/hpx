////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_4D9DE0AC_4362_4DAB_A061_2D3643CE4FCB)
#define HPX_4D9DE0AC_4362_4DAB_A061_2D3643CE4FCB

#include <hpx/hpx.hpp>
#include <hpx/include/async.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <tests/unit/agas/components/server/managed_refcnt_checker.hpp>

namespace hpx { namespace test { namespace stubs
{

struct managed_refcnt_checker
  : components::stub_base<server::managed_refcnt_checker>
{
    static lcos::future<void> take_reference_async(
        naming::id_type const& this_
      , naming::id_type const& gid
        )
    {
        typedef server::managed_refcnt_checker::take_reference_action
            action_type;
        return hpx::async<action_type>(this_, gid);
    }

    static void take_reference(
        naming::id_type const& this_
      , naming::id_type const& gid
        )
    {
        take_reference_async(this_, gid).get();
    }
};

}}}

#endif // HPX_4D9DE0AC_4362_4DAB_A061_2D3643CE4FCB

