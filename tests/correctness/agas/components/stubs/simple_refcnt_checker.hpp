////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_FCB1AFA8_8399_40D9_95DE_A68F861C0CFE)
#define HPX_FCB1AFA8_8399_40D9_95DE_A68F861C0CFE

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <tests/correctness/agas/components/server/simple_refcnt_checker.hpp>

namespace hpx { namespace test { namespace stubs
{

struct simple_refcnt_checker
  : components::stub_base<server::simple_refcnt_checker>
{
    static lcos::promise<void> take_reference_async(
        naming::id_type const& this_
      , naming::id_type const& gid
        )
    {
        typedef server::simple_refcnt_checker::take_reference_action
            action_type;
        return lcos::eager_future<action_type>(this_, gid);
    }

    static void take_reference(
        naming::id_type const& this_
      , naming::id_type const& gid
        )
    {
        take_reference_async(this_, gid).get();
    }

    static lcos::promise<void> garbage_collect_async(
        naming::id_type const& this_
        )
    {
        typedef server::simple_refcnt_checker::garbage_collect_action
            action_type;
        return lcos::eager_future<action_type>(this_);
    }

    static void garbage_collect(
        naming::id_type const& this_
        )
    {
        garbage_collect_async(this_).get();
    }
};

}}}

#endif // HPX_FCB1AFA8_8399_40D9_95DE_A68F861C0CFE

