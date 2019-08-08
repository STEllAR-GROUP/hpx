//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_1FB4A979_B6B6_4845_BD95_3CEC605011A2)
#define HPX_1FB4A979_B6B6_4845_BD95_3CEC605011A2

#include <hpx/config.hpp>
#include <hpx/assertion.hpp>
#include <hpx/errors.hpp>
#include <hpx/lcos/server/object_semaphore.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include <cstdint>

namespace hpx { namespace lcos
{
    template <typename ValueType>
    struct object_semaphore
      : components::client_base<
            object_semaphore<ValueType>,
            lcos::server::object_semaphore<ValueType>
        >
    {
        typedef lcos::server::object_semaphore<ValueType> server_type;

        typedef components::client_base<
            object_semaphore,
            lcos::server::object_semaphore<ValueType>
        > base_type;

        object_semaphore() {}

        object_semaphore(naming::id_type gid) : base_type(gid) {}

        ///////////////////////////////////////////////////////////////////////
        lcos::future<void> signal(launch::async_policy,
            ValueType const& val, std::uint64_t count = 1)
        {
            HPX_ASSERT(this->get_id());
            typedef typename server_type::signal_action action_type;
            return hpx::async<action_type>(this->get_id(), val, count);
        }
        void signal(launch::sync_policy,
            ValueType const& val, std::uint64_t count = 1)
        {
            signal(hpx::async, val, count).get();
        }

        ///////////////////////////////////////////////////////////////////////
        lcos::future<ValueType> get(launch::async_policy)
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
        void abort_pending(launch::sync_policy, error ec = no_success)
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
}}

#endif // HPX_1FB4A979_B6B6_4845_BD95_3CEC605011A2

