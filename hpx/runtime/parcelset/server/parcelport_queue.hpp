//  Copyright (c) 2007-2008 Hartmut Kaiser
//  Copyright (c) 2007 Alexandre (aka Alex) TABBAL
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_SERVER_PARCELPORTQUEUE_MAR_26_2008_1219PM)
#define HPX_PARCELSET_SERVER_PARCELPORTQUEUE_MAR_26_2008_1219PM

#include <list>
#include <boost/thread.hpp>
#include <boost/signal.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/parcelset/parcel.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace server
{
    class parcelport_queue
    {
    public:
        parcelport_queue(hpx::parcelset::parcelport& pp)
          : parcel_port_(pp)
        {}
                
        /// add a new parcel to the end of the parcel queue
        void add_parcel(parcel const& p);
    
//         /// return next available parcel
//         bool get_parcel (parcel& p);
    
        /// register event handler to be notified whenever a parcel arrives
        template <typename F>
        bool register_event_handler(F sink)
        {
            return notify_.connect(sink).connected();
        }

        template <typename F, typename Connection>
        bool register_event_handler(F sink, Connection& conn)
        {
            return (conn = notify_.connect(sink)).connected();
        }

    private:
        // list of pending parcels
//         typedef boost::mutex mutex_type;
//         mutex_type mtx_;
//         
//         typedef std::list<parcel> parcel_list_type;
//         std::list<parcel> parcel_queue_; 

        hpx::parcelset::parcelport& parcel_port_;
        typedef void callback_type(parcelport&, parcel const&);
        boost::signal<callback_type> notify_;
    };

///////////////////////////////////////////////////////////////////////////////
}}}
    
#endif
