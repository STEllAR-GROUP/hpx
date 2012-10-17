////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_E42C5C50_811E_454F_887A_DE7175E72EE9)
#define HPX_E42C5C50_811E_454F_887A_DE7175E72EE9

#include <hpx/hpx_fwd.hpp>

#include <boost/signals2.hpp>
#include <boost/cstdint.hpp>

namespace hpx { namespace parcelset
{
    ///////////////////////////////////////////////////////////////////////////
    // Base class for parcel handler queue, allows to abstract different
    // implementations.
    struct parcelhandler_queue_base
    {
        typedef boost::signals2::scoped_connection connection_type;

        typedef HPX_STD_FUNCTION<void(
            parcelhandler&, naming::gid_type const&
        )> callback_type;

        virtual ~parcelhandler_queue_base() {}

        /// add a new parcel to the end of the parcel queue
        virtual bool add_parcel(parcel const& p) = 0;

        /// add an exception to the parcel queue
        virtual bool add_exception(boost::exception_ptr e) = 0;

        /// return next available parcel
        virtual bool get_parcel(parcel& p, error_code& ec = throws) = 0;

        /// return parcel with given id
        virtual bool get_parcel(parcel& p, naming::gid_type const& parcel_id,
            error_code& ec = throws) = 0;

        /// register event handler to be notified whenever a parcel arrives
        virtual bool register_event_handler(callback_type const& sink) = 0;

        virtual bool register_event_handler(callback_type const& sink,
            connection_type& conn) = 0;

        virtual void set_parcelhandler(parcelhandler* ph) = 0;

        virtual std::size_t get_queue_length() const = 0;
    };
}}

#endif // HPX_E42C5C50_811E_454F_887A_DE7175E72EE9

