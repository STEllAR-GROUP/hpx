//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_QUEUE_FEB_10_2011_1232PM)
#define HPX_LCOS_QUEUE_FEB_10_2011_1232PM

#include <hpx/exception.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <hpx/lcos/stubs/queue.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename ValueType, typename RemoteType = ValueType>
    class queue
      : public components::client_base<queue<ValueType, RemoteType>,
            lcos::stubs::queue<ValueType, RemoteType> >
    {
        typedef components::client_base<
            queue, lcos::stubs::queue<ValueType, RemoteType> > base_type;

    public:
        queue()
        {}

        /// Create a client side representation for the existing
        /// \a server#queue instance with the given global id \a gid.
        queue(naming::id_type gid)
          : base_type(gid)
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        lcos::future<ValueType, RemoteType>
        get_value_async()
        {
            BOOST_ASSERT(this->gid_);
            return this->base_type::get_value_async(this->gid_);
        }

        lcos::future<void>
        set_value_async(RemoteType const& val)
        {
            BOOST_ASSERT(this->gid_);
            RemoteType tmp(val);
            return this->base_type::set_value_async(this->gid_, boost::move(tmp));
        }

        lcos::future<void>
        abort_pending_async(boost::exception_ptr const& e)
        {
            BOOST_ASSERT(this->gid_);
            return this->base_type::abort_pending_async(this->gid_, e);
        }

        ///////////////////////////////////////////////////////////////////////
        ValueType get_value_sync()
        {
            BOOST_ASSERT(this->gid_);
            return this->base_type::get_value_sync(this->gid_);
        }

        void set_value_sync(RemoteType const& val)
        {
            BOOST_ASSERT(this->gid_);
            RemoteType tmp(val);
            this->base_type::set_value_sync(this->gid_, boost::move(tmp));
        }

        void set_value_sync(BOOST_RV_REF(RemoteType) val)
        {
            BOOST_ASSERT(this->gid_);
            this->base_type::set_value_sync(this->gid_, val);
        }

        void abort_pending_sync(boost::exception_ptr const& e)
        {
            this->base_type::abort_pending_sync(this->gid_, e);
        }

        ///////////////////////////////////////////////////////////////////////
        void set_value(RemoteType const& val)
        {
            RemoteType tmp(val);
            this->base_type::set_value(this->gid_, boost::move(tmp));
        }
        void set_value(BOOST_RV_REF(RemoteType) val)
        {
            this->base_type::set_value(this->gid_, val);
        }

        void abort_pending()
        {
            try {
                HPX_THROW_EXCEPTION(no_success, "queue::set_error",
                    "interrupt all pending requests");
            }
            catch (...) {
                this->base_type::abort_pending(this->gid_, boost::current_exception());
            }
        }
    };
}}

#endif

