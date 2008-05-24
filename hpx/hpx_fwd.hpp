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
        // this has to be predeclared to avoid circular header dependencies
        typedef boost::coroutines::shared_coroutine<bool()>::self px_thread_self;
    }

//     namespace components
//     {
//         class component;
//         struct action_base;
// 
//         class factory;        
//         class accumulator;
//         class local_graph;
//         class graph;
//         class vertex;
//     }
}

#endif

