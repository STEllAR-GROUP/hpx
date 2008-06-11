//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_HPX_FWD_MAR_24_2008_1119AM)
#define HPX_HPX_FWD_MAR_24_2008_1119AM

#include <boost/asio.hpp>
#include <boost/coroutine/shared_coroutine.hpp>

namespace hpx
{
    class runtime;
    
    namespace applier
    {
        class applier;
    }
    
    namespace action_manager
    {
        class action_manager;
    }
    
    namespace naming
    {
        struct id_type;
        struct address;
        class locality;
        class resolver_client;
        class resolver_server;

        namespace server
        {
            class reply;
            class request;
        }
    }

    namespace parcelset
    {
        class parcel;
        class parcelport;
        class parcelport_connection;
        class connection_cache;
        class parcelhandler;
        
        namespace server
        {
            class parcelport_queue;
            class parcelport_server_connection;
            class parcelhandler_queue;
        }
    }
    
    namespace threadmanager
    {
        class threadmanager;

        ///////////////////////////////////////////////////////////////////////
        /// \enum thread_state
        ///
        /// The thread_state enumerator encodes the current state of a \a 
        /// px_thread instance
        enum thread_state
        {
            unknown = -1,
            init = 0,       ///< thread is initializing
            depleted = 1,   ///< thread has been depleted (deeply suspended)
            suspended = 2,  ///< thread has been suspended
            pending = 3,    ///< thread is pending (ready to run)
            running = 4,    ///< thread is currently running (active)
            stopped = 5     ///< thread has been stopped an may be garbage collected
        };

        ///////////////////////////////////////////////////////////////////////
        typedef 
            boost::coroutines::shared_coroutine<thread_state()>::self 
        px_thread_self;
        typedef thread_state thread_function_type(px_thread_self&);
    }

    namespace components
    {
        struct action_base;

        class factory;        
        class accumulator;
    }
}

#endif

