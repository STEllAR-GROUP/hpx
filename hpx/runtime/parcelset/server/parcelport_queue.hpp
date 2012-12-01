//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2007 Alexandre (aka Alex) TABBAL
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_SERVER_PARCELPORTQUEUE_MAR_26_2008_1219PM)
#define HPX_PARCELSET_SERVER_PARCELPORTQUEUE_MAR_26_2008_1219PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/parcelset/parcel.hpp>
#include <hpx/performance_counters/parcels/data_point.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace server
{
    class parcelport_queue
    {
    public:
        /// add a new parcel to the end of the parcel queue
        void add_parcel(parcel const& p)
        {
            // do some work (notify event handlers)
            notify_(p);
        }

        /// register event handler to be notified whenever a parcel arrives
        template <typename F>
        void register_event_handler(F sink)
        {
            notify_ = sink;
        }

    private:
        typedef void callback_type(parcel const&);
        HPX_STD_FUNCTION<callback_type> notify_;
    };

///////////////////////////////////////////////////////////////////////////////
}}}

#endif
