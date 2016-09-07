//  Copyright (c) 2013 Shuangyang Yang
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_STUBS_SIMPLE_CENTRAL_TUPLESPACE_MAR_31_2013_0459PM)
#define HPX_STUBS_SIMPLE_CENTRAL_TUPLESPACE_MAR_31_2013_0459PM

#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/runtime/applier/apply.hpp>
#include <hpx/include/async.hpp>

#include "../server/simple_central_tuplespace.hpp"

namespace examples { namespace stubs
{
    ///////////////////////////////////////////////////////////////////////////
    //[simple_central_tuplespace_stubs_inherit
    struct simple_central_tuplespace
      : hpx::components::stub_base<server::simple_central_tuplespace>
    //]
    {
        typedef server::simple_central_tuplespace::tuple_type tuple_type;

        ///////////////////////////////////////////////////////////////////////
        /// put \p tuple into tuplespace.
        ///
        /// \note This function has fire-and-forget semantics. It will not wait
        ///       for the action to be executed. Instead, it will return
        ///       immediately after the action has has been dispatched.
        //[simple_central_tuplespace_stubs_write_async
        static hpx::lcos::future<int> write_async(hpx::naming::id_type const& gid,
            tuple_type const& tuple)
        {
            typedef server::simple_central_tuplespace::write_action action_type;
            return hpx::async<action_type>(gid, tuple);
        }
        //]

        /// put \p tuple into tuplespace.
        ///
        /// \note This function is fully synchronous.
        static int write(hpx::launch::sync_policy,
            hpx::naming::id_type const& gid, tuple_type const& tuple)
        {
            typedef server::simple_central_tuplespace::write_action action_type;
            return hpx::async<action_type>(gid, tuple).get();
        }

        ///////////////////////////////////////////////////////////////////////
        /// read tuple matching \p key from tuplespace within \p timeout.
        ///
        /// \note This function has fire-and-forget semantics. It will not wait
        ///       for the action to be executed. Instead, it will return
        ///       immediately after the action has has been dispatched.
        static hpx::lcos::future<tuple_type>
        read_async(hpx::naming::id_type const& gid, const tuple_type& tp,
            long const timeout)
        {
            typedef server::simple_central_tuplespace::read_action action_type;
            return hpx::async<action_type>(gid, tp, timeout);
        }

        /// read tuple matching within \p timeout.
        ///
        /// \note This function is fully synchronous.
        //[simple_central_tuplespace_stubs_read_sync
        static tuple_type
        read(hpx::launch::sync_policy, hpx::naming::id_type const& gid,
            const tuple_type& tp, long const timeout)
        {
            typedef server::simple_central_tuplespace::read_action action_type;
            return hpx::async<action_type>(gid, tp, timeout).get();
        }
        //]

        ///////////////////////////////////////////////////////////////////////
        /// take tuple matching \p key from tuplespace within \p timeout.
        ///
        /// \returns This function returns an \a hpx::lcos::future. When the
        ///          value of this computation is needed, the get() method of
        ///          the future should be called. If the value is available,
        ///          get() will return immediately; otherwise, it will block
        ///          until the value is ready.
        //[simple_central_tuplespace_stubs_take_async
        static hpx::lcos::future<tuple_type>
        take(hpx::launch::async_policy, hpx::naming::id_type const& gid,
            const tuple_type& tp, long const timeout)
        {
            typedef server::simple_central_tuplespace::take_action action_type;
            return hpx::async<action_type>(gid, tp, timeout);
        }
        //]

        /// take tuple matching \p key from tuplespace within \p timeout.
        ///
        /// \note This function is fully synchronous.
        static tuple_type take(hpx::launch::sync_policy,
            hpx::naming::id_type const& gid, const tuple_type& tp,
            const long timeout)
        {
            // The following get yields control while the action is executed.
            return take(hpx::launch::async, gid, tp, timeout).get();
        }
    };
}}

#endif

