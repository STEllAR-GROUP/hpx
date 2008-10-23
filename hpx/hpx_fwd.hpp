//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_HPX_FWD_MAR_24_2008_1119AM)
#define HPX_HPX_FWD_MAR_24_2008_1119AM

#include <boost/config.hpp>
#if defined(BOOST_WINDOWS)
#include <winsock2.h>
#endif

#include <boost/asio.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/coroutine/shared_coroutine.hpp>
#include <hpx/config.hpp>

/// \namespace hpx
///
/// The namespace \a hpx is the main namespace of the HPX library. All classes
/// functions and variables are defined inside this namespace.
namespace hpx
{
    class HPX_EXPORT runtime;

    /// \namespace applier
    ///
    /// The namespace \a applier contains all definitions needed for the
    /// class \a hpx#applier#applier and its related functionality. This 
    /// namespace is part of the HPX core module.
    namespace applier
    {
        class HPX_EXPORT applier;
    }

    /// \namespace actions
    ///
    /// The namespace \a actions contains all definitions needed for the
    /// class \a hpx#action_manager#action_manager and its related 
    /// functionality. This namespace is part of the HPX core module.
    namespace actions
    {
        struct HPX_EXPORT base_action;
        typedef boost::shared_ptr<base_action> action_type;

        class HPX_EXPORT continuation;
        typedef boost::shared_ptr<continuation> continuation_type;

        class HPX_EXPORT action_manager;
    }

    /// \namespace naming
    ///
    /// The namespace \a naming contains all definitions needed for the DGAS
    /// (Distributed Global Address Space) service.
    namespace naming
    {
        struct HPX_EXPORT id_type;
        struct HPX_EXPORT address;
        class HPX_EXPORT locality;
        class HPX_EXPORT resolver_client;
        class HPX_EXPORT resolver_server;

        namespace server
        {
            class reply;
            class request;
        }
    }

    /// \namespace parcelset
    namespace parcelset
    {
        class HPX_EXPORT parcel;
        class HPX_EXPORT parcelport;
        class parcelport_connection;
        class connection_cache;
        class HPX_EXPORT parcelhandler;
        
        namespace server
        {
            class parcelport_queue;
            class parcelport_server_connection;
            class parcelhandler_queue;
        }
    }

    /// \namespace threads
    ///
    /// The namespace \a threadmanager contains all the definitions required 
    /// for the scheduling, execution and general management of \a 
    /// hpx#threadmanager#thread's.
    namespace threads
    {
        class HPX_EXPORT thread;
        class HPX_EXPORT threadmanager;

        ///////////////////////////////////////////////////////////////////////
        /// \enum thread_state
        ///
        /// The thread_state enumerator encodes the current state of a \a 
        /// thread instance
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
        HPX_EXPORT char const* const get_thread_state_name(thread_state state);

        ///////////////////////////////////////////////////////////////////////
        typedef 
            boost::coroutines::shared_coroutine<thread_state()>
        coroutine_type;
        typedef coroutine_type::thread_id_type thread_id_type;
        typedef coroutine_type::self thread_self;
        typedef thread_state thread_function_type(thread_self&);

        ///////////////////////////////////////////////////////////////////////
        /// \brief  Set the thread state of the \a thread referenced by the 
        ///         thread_id \a id.
        ///
        /// \param self       [in] A reference to the \a thread executing 
        ///                   this function. 
        /// \param id         [in] The thread id of the thread the state should 
        ///                   be modified for.
        /// \param newstate   [in] The new state to be set for the thread 
        ///                   referenced by the \a id parameter.
        ///
        /// \returns          This function returns the previous state of the 
        ///                   thread referenced by the \a id parameter. It will 
        ///                   return one of the values as defined by the 
        ///                   \a thread_state enumeration. If the 
        ///                   thread is not known to the threadmanager the 
        ///                   return value will be \a thread_state#unknown.
        ///
        /// \note             This function yields the \a thread specified 
        ///                   by the parameter \a self if the thread referenced 
        ///                   by  the parameter \a id is in \a 
        ///                   thread_state#active state.
        HPX_EXPORT thread_state 
            set_thread_state(thread_self& self, thread_id_type id, 
                thread_state newstate);

        ///////////////////////////////////////////////////////////////////////
        /// \brief  Set the thread state of the \a thread referenced by the 
        ///         thread_id \a id.
        ///
        /// \param id         [in] The thread id of the thread the state should 
        ///                   be modified for.
        /// \param newstate   [in] The new state to be set for the thread 
        ///                   referenced by the \a id parameter.
        ///
        /// \returns          This function returns the previous state of the 
        ///                   thread referenced by the \a id parameter. It will 
        ///                   return one of the values as defined by the 
        ///                   \a thread_state enumeration. If the 
        ///                   thread is not known to the threadmanager the 
        ///                   return value will be \a thread_state#unknown.
        ///
        /// \note             If the thread referenced by the parameter \a id 
        ///                   is in \a thread_state#active state this function 
        ///                   does nothing except returning 
        ///                   thread_state#unknown. 
        HPX_EXPORT thread_state 
            set_thread_state(thread_id_type id, thread_state newstate);

        ///////////////////////////////////////////////////////////////////////
        /// \brief  Set the thread state of the \a thread referenced by the 
        ///         thread_id \a id.
        ///
        /// \param id         [in] The thread id of the thread the state should 
        ///                   be modified for.
        /// \param newstate   [in] The new state to be set for the thread 
        ///                   referenced by the \a id parameter.
        /// \param at_time
        ///
        /// \returns
        HPX_EXPORT thread_id_type 
            set_thread_state(thread_id_type id, thread_state newstate, 
                boost::posix_time::ptime const& at_time);

        ///////////////////////////////////////////////////////////////////////
        /// \brief  Set the thread state of the \a thread referenced by the 
        ///         thread_id \a id.
        ///
        /// \param id         [in] The thread id of the thread the state should 
        ///                   be modified for.
        /// \param newstate   [in] The new state to be set for the thread 
        ///                   referenced by the \a id parameter.
        /// \param after_duration
        ///                   
        ///
        /// \returns
        ///////////////////////////////////////////////////////////////////////
        HPX_EXPORT thread_id_type 
            set_thread_state(thread_id_type id, thread_state newstate, 
                boost::posix_time::time_duration const& after_duration);

    }

    /// \namespace components
    namespace components
    {
        namespace detail 
        { 
            struct this_type {};
            struct simple_component_tag {};
            struct managed_component_tag {};
        }

        template <typename Component> 
        class simple_component_base;

        template <typename Component, typename Derived = detail::this_type>
        class managed_component;

        struct HPX_EXPORT component_factory_base;

        template <typename Component> 
        struct component_factory;

        class runtime_support;
        class memory;
        class memory_block;

        namespace stubs 
        {
            class runtime_support;
            class memory;
            class memory_block;
        }

        namespace server
        {
            class HPX_EXPORT runtime_support;
            class memory;
            class memory_block;
        }
    }

    /// \namespace lcos
    namespace lcos
    {
        class base_lco;
        template <typename Result> 
        class base_lco_with_value;

        template <typename Result, int N = 1> 
        class future_value;

        template <typename Action, typename Result,
            typename DirectExecute = typename Action::direct_execution> 
        class eager_future;

        template <typename Action, typename Result,
            typename DirectExecute = typename Action::direct_execution> 
        class lazy_future;
    }

    /// \namespace util
    namespace util
    {
        class HPX_EXPORT section;
    }
}

#endif

