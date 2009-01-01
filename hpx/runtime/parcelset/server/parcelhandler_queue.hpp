//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_SERVER_PARCELHANDLER_QUEUE_MAY_17_2008_1222PM)
#define HPX_PARCELSET_SERVER_PARCELHANDLER_QUEUE_MAY_17_2008_1222PM

#include <list>
#include <boost/thread.hpp>
#include <boost/signal.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/parcelset/parcel.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace server
{
    class parcelhandler_queue
    {
    public:
        parcelhandler_queue(parcelhandler& ph)
          : parcelhandler_(ph)
        {}
                
        /// add a new parcel to the end of the parcel queue
        void add_parcel(parcel const& p);
    
        /// return next available parcel
        bool get_parcel (parcel& p);
    
        /// return parcel based on the destination component type that will be 
        /// invoked
        bool get_parcel (components::component_type c, parcel& p);

        /// return parcel of tag based on parcel id
        bool get_parcel (parcel_id tag, parcel& p);
    
        /// return parcel of from the given source locality 
        bool get_parcel_from (naming::id_type src, parcel& p);
    
        /// return parcel for destination 'dest' 
        bool get_parcel_for (naming::id_type dest, parcel& p);
        
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
        typedef boost::mutex mutex_type;
        mutex_type mtx_;
        
        typedef std::list<parcel> parcel_list_type;
        std::list<parcel> parcel_queue_; 

        parcelhandler& parcelhandler_;
        
        typedef void callback_type(parcelhandler&, naming::address const&);
        boost::signal<callback_type> notify_;
    };

///////////////////////////////////////////////////////////////////////////////
}}}
    
#endif
