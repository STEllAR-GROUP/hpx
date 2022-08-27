//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_distributed/async.hpp>
#include <hpx/components/client_base.hpp>
#include <hpx/lcos_distributed/server/object_semaphore.hpp>
#include <hpx/modules/errors.hpp>

#include <cstdint>
#include <utility>

namespace hpx { namespace lcos {

    template <typename ValueType>
    struct object_semaphore
      : components::client_base<object_semaphore<ValueType>,
            lcos::server::object_semaphore<ValueType>>
    {
        using server_type = lcos::server::object_semaphore<ValueType>;
        using base_type = components::client_base<object_semaphore,
            lcos::server::object_semaphore<ValueType>>;

        object_semaphore() = default;

        explicit object_semaphore(hpx::id_type gid)
          : base_type(HPX_MOVE(gid))
        {
        }

        ///////////////////////////////////////////////////////////////////////
        hpx::future<void> signal(
            launch::async_policy, ValueType const& val, std::uint64_t count = 1)
        {
            HPX_ASSERT(this->get_id());
            typedef typename server_type::signal_action action_type;
            return hpx::async<action_type>(this->get_id(), val, count);
        }

        void signal(
            launch::sync_policy, ValueType const& val, std::uint64_t count = 1)
        {
            signal(hpx::launch::async, val, count).get();
        }

        ///////////////////////////////////////////////////////////////////////
        hpx::future<ValueType> get(launch::async_policy)
        {
            HPX_ASSERT(this->get_id());
            typedef typename server_type::get_action action_type;
            return hpx::async<action_type>(this->get_id());
        }

        ValueType get(launch::sync_policy)
        {
            return get(launch::async).get();
        }

        ///////////////////////////////////////////////////////////////////////
        future<void> abort_pending(launch::async_policy, error ec = no_success)
        {
            HPX_ASSERT(this->get_id());
            typedef typename server_type::abort_pending_action action_type;
            return hpx::async<action_type>(this->get_id(), ec);
        }

        void abort_pending(launch::sync_policy, error = no_success)
        {
            abort_pending(launch::async).get();
        }

        ///////////////////////////////////////////////////////////////////////
        void wait(launch::async_policy)
        {
            HPX_ASSERT(this->get_id());
            typedef typename server_type::wait_action action_type;
            return hpx::async<action_type>(this->get_id());
        }

        void wait(launch::sync_policy)
        {
            wait(launch::async).get();
        }
    };
}}    // namespace hpx::lcos
