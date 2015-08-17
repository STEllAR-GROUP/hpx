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
#include <hpx/include/client.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>

#include <tests/unit/agas/components/stubs/managed_refcnt_checker.hpp>

namespace hpx { namespace test
{

struct managed_refcnt_monitor
  : components::client_base<
        managed_refcnt_monitor
      , stubs::managed_refcnt_checker
    >
{
    typedef components::client_base<
        managed_refcnt_monitor
      , stubs::managed_refcnt_checker
    > base_type;

  private:
    lcos::promise<void> flag_;
    naming::id_type const locality_;

    using base_type::create;

  public:
    typedef server::managed_refcnt_checker server_type;

    /// Create a new component on the target locality.
    explicit managed_refcnt_monitor(
        naming::gid_type const& locality
        )
      : locality_(naming::get_locality_from_gid(locality)
                , naming::id_type::unmanaged)
    {
        gid_ = stub_type::create_async(locality_, flag_.get_id());
    }

    /// Create a new component on the target locality.
    explicit managed_refcnt_monitor(
        naming::id_type const& locality
        )
      : locality_(naming::get_locality_from_id(locality))
    {
        gid_ = stub_type::create_async(locality_, flag_.get_id());
    }

    lcos::future<void> take_reference_async(
        naming::id_type const& gid
        )
    {
        return this->base_type::take_reference_async(get_id(), gid);
    }

    void take_reference(
        naming::id_type const& gid
        )
    {
        return this->base_type::take_reference(get_id(), gid);
    }

    bool is_ready()
    {
        // Flush pending reference counting operations on the target locality.
        agas::garbage_collect(locality_);

        return flag_.is_ready();
    }

    template <
        typename Duration
    >
    bool is_ready(
        Duration const& d
        )
    {
        // Flush pending reference counting operations on the target locality.
        agas::garbage_collect(locality_);

        // Schedule a wakeup.
        threads::set_thread_state(threads::get_self_id(), d, threads::pending);

        // Suspend this pxthread.
        threads::get_self().yield(threads::suspended);

        return flag_.is_ready();
    }
};

struct managed_object
  : components::client_base<
        managed_object
      , stubs::managed_refcnt_checker
    >
{
    typedef components::client_base<
        managed_object
      , stubs::managed_refcnt_checker
    > base_type;

  private:
    using base_type::create;

  public:
    typedef server::managed_refcnt_checker server_type;

    /// Create a new component on the target locality.
    explicit managed_object(
        naming::gid_type const& locality
        )
    {
        gid_ = stub_type::create_async(
            naming::id_type(locality, naming::id_type::unmanaged),
            naming::invalid_id);
    }

    /// Create a new component on the target locality.
    explicit managed_object(
        naming::id_type const& locality
        )
    {
        gid_ = stub_type::create_async(locality, naming::invalid_id);
    }
};

}}

#endif // HPX_901997BE_9730_41F7_9DBC_AD1DC70D7819

