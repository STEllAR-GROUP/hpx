//  Copyright (c) 2007 Alexandre (aka Alex) TABBAL, & Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(DISTPX_PARCELQUEUE_JUL_02_2007_0228PM)
#define DISTPX_PARCELQUEUE_JUL_02_2007_0228PM

#include <list>
#include <boost/thread.hpp>

#include <distpx/parcelset/parcel.hpp>
#include <distpx/threadmanager/threadmanager.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace distpx { namespace parcelset { namespace server
{
    class parcel_queue
    {
    private:
        // list of pending parcels
        typedef boost::mutex mutex_type;
        mutex_type mtx_;
        
        typedef std::list<parcel> parcel_list_type;
        std::list<parcel> parcel_queue_; 

        threadmanager::threadmanager& thread_manager_;
        
    public:
        parcel_queue(threadmanager::threadmanager& thread_manager)
          : thread_manager_(thread_manager)
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
    };

///////////////////////////////////////////////////////////////////////////////
}}}
    
#endif
