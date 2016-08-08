//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_QUEUE_FEB_10_2011_1232PM)
#define HPX_LCOS_QUEUE_FEB_10_2011_1232PM

#include <hpx/config.hpp>
#include <hpx/lcos/server/queue.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include <boost/exception_ptr.hpp>

#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename ValueType, typename RemoteType = ValueType>
    class queue
      : public components::client_base<
            queue<ValueType, RemoteType>,
            lcos::server::queue<ValueType, RemoteType>
        >
    {
        typedef components::client_base<
                queue, lcos::server::queue<ValueType, RemoteType>
            > base_type;

    public:
        queue()
        {}

        /// Create a client side representation for the existing
        /// \a server#queue instance with the given global id \a gid.
        queue(future<id_type> && gid)
          : base_type(std::move(gid))
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component
        future<ValueType> get_value()
        {
            typedef typename
                lcos::base_lco_with_value<ValueType, RemoteType>::get_value_action
            action_type;

            HPX_ASSERT(this->get_gid());
            return hpx::async<action_type>(this->get_gid());
        }

        future<void> set_value(RemoteType && val)
        {
            typedef typename
                lcos::base_lco_with_value<ValueType, RemoteType>::set_value_action
            action_type;

            HPX_ASSERT(this->get_gid());
            return hpx::async<action_type>(this->get_gid(), std::move(val));
        }

        future<void> set_value(RemoteType val)
        {
            typedef typename
                lcos::base_lco_with_value<ValueType, RemoteType>::set_value_action
            action_type;

            HPX_ASSERT(this->get_gid());
            return hpx::async<action_type>(this->get_gid(), std::move(val));
        }

        future<void> abort_pending()
        {
            typedef lcos::base_lco::set_exception_action action_type;

            HPX_ASSERT(this->get_gid());
            boost::exception_ptr exception =
                HPX_GET_EXCEPTION(hpx::no_success, "queue::abort_pending", "");
            return hpx::async<action_type>(this->get_gid(), exception);
        }

        ///////////////////////////////////////////////////////////////////////
        ValueType get_value(launch::sync_policy)
        {
            return get_value().get();
        }

        void set_value(launch::sync_policy, RemoteType const& val)
        {
            set_value(val).get();
        }

        void set_value(launch::sync_policy, RemoteType && val) //-V659
        {
            set_value(std::move(val)).get();
        }

        void abort_pending(launch::sync_policy)
        {
            abort_pending().get();
        }

#if defined(HPX_HAVE_ASYNC_FUNCTION_COMPATIBILITY)
        ValueType get_value_sync()
        {
            return get_value(launch::sync);
        }

        void set_value_sync(RemoteType const& val)
        {
            set_value(launch::sync, val);
        }

        void set_value_sync(RemoteType && val) //-V659
        {
            set_value(launch::sync, std::move(val));
        }

        void abort_pending_sync()
        {
            abort_pending(launch::sync);
        }
#endif
    };
}}

#endif

