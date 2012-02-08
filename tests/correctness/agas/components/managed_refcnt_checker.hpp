////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_901997BE_9730_41F7_9DBC_AD1DC70D7819)
#define HPX_901997BE_9730_41F7_9DBC_AD1DC70D7819

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/promise.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>

#include <tests/correctness/agas/components/stubs/managed_refcnt_checker.hpp>

namespace hpx { namespace test
{

struct managed_refcnt_checker
  : components::client_base<
        managed_refcnt_checker
      , stubs::managed_refcnt_checker
    >
{
    typedef components::client_base<
        managed_refcnt_checker
      , stubs::managed_refcnt_checker
    > base_type;

  private:
    lcos::promise<void> flag_;

    using base_type::create;
    using base_type::create_one;

  public:
    typedef server::managed_refcnt_checker server_type;

    /// Create a new component on the target locality.
    explicit managed_refcnt_checker(
        naming::gid_type const& locality
        )
    {
        this->base_type::create_one(locality, flag_.get_gid());
    }

    /// Create a new component on the target locality.
    explicit managed_refcnt_checker(
        naming::id_type const& locality
        )
    {
        this->base_type::create_one(locality, flag_.get_gid());
    }

    lcos::promise<void> take_reference_async(
        naming::id_type const& gid
        )
    {
        BOOST_ASSERT(gid_);
        return this->base_type::take_reference_async(gid_, gid);
    }

    void take_reference(
        naming::id_type const& gid
        )
    {
        BOOST_ASSERT(gid_);
        return this->base_type::take_reference(gid_, gid);
    }

    bool ready()
    {
        // Do a round of garbage collection on the target.
        this->base_type::garbage_collect(gid_);

        return flag_.ready();
    }

    template <
        typename Duration
    >
    bool ready(
        Duration const& d
        )
    {
        // Do a round of garbage collection on the target.
        this->base_type::garbage_collect(gid_);

        // Schedule a wakeup.
        threads::set_thread_state(threads::get_self_id(), d, threads::pending);

        // Suspend this pxthread.
        threads::get_self().yield(threads::suspended);

        return flag_.ready();
    }
};

}}

#endif // HPX_901997BE_9730_41F7_9DBC_AD1DC70D7819

