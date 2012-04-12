//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_STUBS_QUEUE_FEB_10_2011_1132AM)
#define HPX_LCOS_STUBS_QUEUE_FEB_10_2011_1132AM

#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/lcos/async.hpp>
#include <hpx/lcos/server/queue.hpp>
#include <hpx/traits/promise_remote_result.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace stubs
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename ValueType, typename RemoteType = ValueType>
    struct queue
      : public components::stubs::stub_base<
            lcos::server::queue<ValueType, RemoteType> >
    {
        ///////////////////////////////////////////////////////////////////////
        static lcos::future<ValueType, RemoteType>
        get_value_async(naming::id_type const& gid)
        {
            typedef typename
                lcos::base_lco_with_value<ValueType, RemoteType>::get_value_action
            action_type;
            return lcos::async<action_type>(gid);
        }

        static lcos::future<void>
        set_value_async(naming::id_type const& gid, BOOST_RV_REF(RemoteType) val)
        {
            typedef typename
                lcos::base_lco_with_value<ValueType, RemoteType>::set_value_action
            action_type;
            return lcos::async<action_type>(gid, boost::move(val));
        }

        static lcos::future<void>
        abort_pending_async(naming::id_type const& gid,
            boost::exception_ptr const& e)
        {
            typedef lcos::base_lco::set_exception_action action_type;
            return lcos::async<action_type>(gid, e);
        }

        ///////////////////////////////////////////////////////////////////////
        static ValueType get_value_sync(naming::id_type const& gid)
        {
            return get_value_async(gid).get();
        }

        static void set_value_sync(naming::id_type const& gid, BOOST_RV_REF(RemoteType) val)
        {
            set_value_async(gid, boost::move(val)).get();
        }

        static void abort_pending_sync(naming::id_type const& gid,
            boost::exception_ptr const& e)
        {
            abort_pending_async(gid, e).get();
        }

        ///////////////////////////////////////////////////////////////////////
        static void set_value(naming::id_type const& gid, BOOST_RV_REF(RemoteType) val)
        {
            typedef typename
                lcos::base_lco_with_value<ValueType, RemoteType>::set_value_action
            action_type;
            hpx::applier::apply<action_type>(gid, val);
        }

        static void abort_pending(naming::id_type const& gid,
            boost::exception_ptr const& e)
        {
            typedef lcos::base_lco::set_exception_action action_type;
            hpx::applier::apply<action_type>(gid, e);
        }
    };
}}}

#endif

