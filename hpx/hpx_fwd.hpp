//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_HPX_FWD_MAR_24_2008_1119AM)
#define HPX_HPX_FWD_MAR_24_2008_1119AM

#include <boost/asio.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/coroutine/shared_coroutine.hpp>
#include <hpx/config.hpp>

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
        class px_thread;
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
            active = 1,     ///< thread is currently active (running,
                            ///< has resources)
            pending = 2,    ///< thread is pending (ready to run, but 
                            ///< no hardware resource available)
            suspended = 3,  ///< thread has been suspended (waiting for 
                            ///< synchronization event, but still known 
                            ///< and under control of the threadmanager)
            depleted = 4,   ///< thread has been depleted (deeply 
                            ///< suspended, it is not known to the thread 
                            ///< manager)
            terminated = 5  ///< thread has been stopped an may be garbage 
                            ///< collected
        };

        ///////////////////////////////////////////////////////////////////////
        typedef 
            boost::coroutines::shared_coroutine<thread_state()>
        coroutine_type;
        typedef coroutine_type::thread_id_type thread_id_type;
        typedef coroutine_type::self px_thread_self;
        typedef thread_state thread_function_type(px_thread_self&);
    }

    namespace components
    {
        struct action_base;
        typedef boost::shared_ptr<action_base> action_type;

        class continuation;
        typedef boost::shared_ptr<continuation> continuation_type;

        class runtime_support;

        class accumulator;
        class distributing_factory;

        namespace stubs 
        {
            class runtime_support;
            class accumulator;
            class distributing_factory;
        }

        namespace server
        {
            class runtime_support;
            class memory;
            class accumulator;
            class distributing_factory;
        }
    }

    namespace lcos
    {
        struct base_lco;
        template <typename Result> struct base_lco_with_value;
        template <typename Result> class simple_future;
        template <
            typename Action, typename Result,
            typename DirectExecute = typename Action::direct_execution
        > class eager_future;
        template <typename Action, typename Result> class lazy_future;
    }
}

#endif

