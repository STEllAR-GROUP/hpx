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
        parcelport_queue()
        {}

        /// add a new parcel to the end of the parcel queue
        void add_parcel(boost::shared_ptr<std::vector<parcel> > data,
            threads::thread_priority priority,
            performance_counters::parcels::data_point const& receive_data)
        {
            // do some work (notify event handlers)
            notify_(data, priority, receive_data);
        }

        void add_exception(boost::exception_ptr e)
        {
            notify_error_(e);
        }

        /// register event handler to be notified whenever a parcel arrives
        template <typename F, typename Error>
        void register_event_handlers(F sink, Error error_sink)
        {
            notify_ = sink;
            notify_error_ = error_sink;
        }

    private:
        typedef void callback_type(boost::shared_ptr<std::vector<parcel> >, 
            threads::thread_priority,
            performance_counters::parcels::data_point const&);
        typedef void error_callback_type(boost::exception_ptr);

        HPX_STD_FUNCTION<callback_type> notify_;
        HPX_STD_FUNCTION<error_callback_type> notify_error_;
    };
}}}

#endif
