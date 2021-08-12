////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/async_distributed/promise.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/client.hpp>
#include <hpx/threading_base/thread_data.hpp>
#include <hpx/threading_base/thread_helpers.hpp>

#include "stubs/simple_refcnt_checker.hpp"

#include <utility>

namespace hpx { namespace test {

    struct simple_refcnt_monitor
      : components::client_base<simple_refcnt_monitor,
            stubs::simple_refcnt_checker>
    {
        typedef components::client_base<simple_refcnt_monitor,
            stubs::simple_refcnt_checker>
            base_type;

    private:
        lcos::promise<void> flag_promise_;
        lcos::future<void> flag_;
        naming::id_type const locality_;

    public:
        typedef server::simple_refcnt_checker server_type;

        simple_refcnt_monitor(hpx::future<id_type>&& id)
          : base_type(std::move(id))
        {
        }

        /// Create a new component on the target locality.
        explicit simple_refcnt_monitor(naming::gid_type const& locality)
          : base_type()
          , flag_promise_()
          , flag_(flag_promise_.get_future())
          , locality_(naming::get_locality_from_gid(locality),
                naming::id_type::unmanaged)
        {
            static_cast<base_type&>(*this) =
                hpx::new_<server::simple_refcnt_checker>(
                    locality_, flag_promise_.get_id());
        }

        /// Create a new component on the target locality.
        explicit simple_refcnt_monitor(naming::id_type const& locality)
          : base_type()
          , flag_promise_()
          , flag_(flag_promise_.get_future())
          , locality_(naming::get_locality_from_id(locality))
        {
            static_cast<base_type&>(*this) =
                hpx::new_<server::simple_refcnt_checker>(
                    locality_, flag_promise_.get_id());
        }

        lcos::future<void> take_reference_async(naming::id_type const& gid)
        {
            return this->base_type::take_reference_async(get_id(), gid);
        }

        void take_reference(naming::id_type const& gid)
        {
            return this->base_type::take_reference(get_id(), gid);
        }

        bool is_ready()
        {
            // Flush pending reference counting operations on the target locality.
            agas::garbage_collect(locality_);

            return flag_.is_ready();
        }

        template <typename Duration>
        bool is_ready(Duration const& d)
        {
            // Flush pending reference counting operations on the target locality.
            agas::garbage_collect(locality_);

            // keep ourselves alive
            threads::thread_id_ref_type self_id = threads::get_self_id();

            // Schedule a wakeup.
            threads::set_thread_state(
                self_id.noref(), d, threads::thread_schedule_state::pending);

            // Suspend this thread.
            threads::get_self().yield(threads::thread_result_type(
                threads::thread_schedule_state::suspended,
                threads::invalid_thread_id));

            return flag_.is_ready();
        }
    };

    struct simple_object
      : components::client_base<simple_object, stubs::simple_refcnt_checker>
    {
        typedef components::client_base<simple_object,
            stubs::simple_refcnt_checker>
            base_type;

    public:
        typedef server::simple_refcnt_checker server_type;

        /// Create a new component on the target locality.
        explicit simple_object(naming::gid_type const& locality)
          : base_type(hpx::new_<server::simple_refcnt_checker>(
                naming::id_type(locality, naming::id_type::unmanaged),
                naming::invalid_id))
        {
        }

        /// Create a new component on the target locality.
        explicit simple_object(naming::id_type const& locality)
          : base_type(hpx::new_<server::simple_refcnt_checker>(
                locality, naming::invalid_id))
        {
        }
    };

}}    // namespace hpx::test

#endif
