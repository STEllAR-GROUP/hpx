//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_1FB4A979_B6B6_4845_BD95_3CEC605011A2)
#define HPX_1FB4A979_B6B6_4845_BD95_3CEC605011A2

#include <hpx/config.hpp>
#include <hpx/exception_fwd.hpp>
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
#if defined(HPX_HAVE_ASYNC_FUNCTION_COMPATIBILITY)
        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        lcos::future<void> signal_async(ValueType const& val,
            std::uint64_t count = 1)
        {
            return signal(launch::async, val, count);
        }
        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        void signal_sync(ValueType const& val, std::uint64_t count = 1)
        {
            signal(launch::sync, val, count);
        }
        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        void signal(ValueType const& val, std::uint64_t count = 1)
        {
            signal(launch::sync, val, count);
        }
#endif

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
#if defined(HPX_HAVE_ASYNC_FUNCTION_COMPATIBILITY)
        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        lcos::future<ValueType> get_async()
        {
            return get(launch::async);
        }
        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        ValueType get_sync()
        {
            get(launch::sync);
        }
        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        ValueType get()
        {
            get(launch::sync);
        }
#endif

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
#if defined(HPX_HAVE_ASYNC_FUNCTION_COMPATIBILITY)
        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        future<void> abort_pending_async(error ec = no_success)
        {
            HPX_ASSERT(this->get_id());
            return this->base_type::abort_pending_async(this->get_id(), ec);
        }
        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        void abort_pending_sync(error ec = no_success)
        {
            HPX_ASSERT(this->get_id());
            this->base_type::abort_pending_sync(this->get_id(), ec);
        }
        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        void abort_pending(error ec = no_success)
        {
            abort_pending_sync(ec);
        }
#endif

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
#if defined(HPX_HAVE_ASYNC_FUNCTION_COMPATIBILITY)
        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        future<void> wait_async()
        {
            return wait(launch::async);
        }
        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        void wait_sync()
        {
            wait(launch::sync);
        }
        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        void wait()
        {
            wait(launch::sync);
        }
#endif
    };
}}

#endif // HPX_1FB4A979_B6B6_4845_BD95_3CEC605011A2

