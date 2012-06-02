//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx_fwd.hpp

#if !defined(HPX_HPX_FWD_MAR_24_2008_1119AM)
#define HPX_HPX_FWD_MAR_24_2008_1119AM

#include <cstdlib>
#include <vector>

#include <boost/config.hpp>
#include <boost/version.hpp>

#if BOOST_VERSION < 104200
// Please update your Boost installation (see www.boost.org for details).
#error HPX cannot be compiled with a Boost version earlier than V1.42.
#endif

#if defined(BOOST_WINDOWS)
#include <winsock2.h>
#include <windows.h>
#endif

#include <boost/shared_ptr.hpp>
#include <boost/intrusive_ptr.hpp>
#include <boost/cstdint.hpp>
#include <boost/coroutine/coroutine.hpp>
#include <boost/detail/scoped_enum_emulation.hpp>

#include <hpx/config.hpp>
#include <hpx/config/function.hpp>
#include <hpx/util/unused.hpp>
#include <hpx/runtime/threads/detail/tagged_thread_state.hpp>
#include <hpx/traits.hpp>

/// \namespace hpx
///
/// The namespace \a hpx is the main namespace of the HPX library. All classes
/// functions and variables are defined inside this namespace.
namespace hpx
{
    /// \cond NOINTERNAL
    class error_code;

    HPX_EXCEPTION_EXPORT extern error_code throws;

    /// \namespace applier
    ///
    /// The namespace \a applier contains all definitions needed for the
    /// class \a hpx#applier#applier and its related functionality. This
    /// namespace is part of the HPX core module.
    namespace applier
    {
        class HPX_API_EXPORT applier;

        /// The function \a get_applier returns a reference to the (thread
        /// specific) applier instance.
        HPX_API_EXPORT applier& get_applier();
        HPX_API_EXPORT applier* get_applier_ptr();
    }

    namespace agas
    {
        struct HPX_API_EXPORT addressing_service;

        enum service_mode
        {
            service_mode_invalid = -1,
            service_mode_bootstrap = 0,
            service_mode_hosted = 1
        };
    }

    /// \namespace naming
    ///
    /// The namespace \a naming contains all definitions needed for the AGAS
    /// (Active Global Address Space) service.
    namespace naming
    {
        typedef agas::addressing_service resolver_client;

        struct HPX_API_EXPORT gid_type;
        struct HPX_API_EXPORT id_type;
        struct HPX_API_EXPORT address;
        class HPX_API_EXPORT locality;

        HPX_API_EXPORT resolver_client& get_agas_client();
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \namespace parcelset
    namespace parcelset
    {
        class HPX_API_EXPORT parcel;
        class HPX_API_EXPORT parcelport;
        class parcelport_connection;
        class HPX_API_EXPORT parcelhandler;

        namespace server
        {
            class parcelport_queue;
            class parcelport_server_connection;

            struct parcelhandler_queue_base;

            namespace policies
            {
                struct global_parcelhandler_queue;

                typedef global_parcelhandler_queue parcelhandler_queue;
            }
        }
    }

    /// \namespace threads
    ///
    /// The namespace \a threadmanager contains all the definitions required
    /// for the scheduling, execution and general management of \a
    /// hpx#threadmanager#thread's.
    namespace threads
    {
        namespace policies
        {
#if defined(HPX_GLOBAL_SCHEDULER)
            class HPX_API_EXPORT global_queue_scheduler;
#endif
#if defined(HPX_LOCAL_SCHEDULER)
            class HPX_API_EXPORT local_queue_scheduler;
#endif
#if defined(HPX_ABP_SCHEDULER)
            struct HPX_API_EXPORT abp_queue_scheduler;
#endif
#if defined(HPX_ABP_PRIORITY_SCHEDULER)
            class HPX_API_EXPORT abp_priority_queue_scheduler;
#endif

            class HPX_API_EXPORT local_priority_queue_scheduler;

#if defined(HPX_HIERARCHY_SCHEDULER)
            class HPX_API_EXPORT hierarchy_scheduler;
#endif
#if defined(HPX_PERIODIC_PRIORITY_SCHEDULER)
            class HPX_API_EXPORT periodic_priority_scheduler;
#endif

            class HPX_API_EXPORT callback_notifier;

            // define the default scheduler to use
            typedef local_priority_queue_scheduler queue_scheduler;
        }

        struct HPX_API_EXPORT threadmanager_base;
        class HPX_API_EXPORT thread_data;

        template <
            typename SchedulingPolicy,
            typename NotificationPolicy = threads::policies::callback_notifier>
        class HPX_API_EXPORT threadmanager_impl;

        /// \enum thread_state_enum
        ///
        /// The \a thread_state_enum enumerator encodes the current state of a
        /// \a thread instance
        enum thread_state_enum
        {
            unknown = 0,
            active = 1,         /*!< thread is currently active (running,
                                     has resources) */
            pending = 2,        /*!< thread is pending (ready to run, but
                                     no hardware resource available) */
            suspended = 3,      /*!< thread has been suspended (waiting for
                                     synchronization event, but still
                                     known and under control of the
                                     threadmanager) */
            depleted = 4,       /*!< thread has been depleted (deeply
                                     suspended, it is not known to the
                                     thread manager) */
            terminated = 5      /*!< thread has been stopped an may be
                                     garbage collected */
        };

        /// \enum thread_priority
        enum thread_priority
        {
            thread_priority_default = 0,      ///< use default priority
            thread_priority_low = 1,          ///< low thread priority
            thread_priority_normal = 2,       ///< normal thread priority (default)
            thread_priority_critical = 3      ///< high thread priority
        };

        typedef threads::detail::tagged_thread_state<thread_state_enum> thread_state;

        HPX_API_EXPORT char const* get_thread_state_name(thread_state_enum state);
        HPX_API_EXPORT char const* get_thread_priority_name(thread_priority priority);

        /// \enum thread_state_ex_enum
        ///
        /// The \a thread_state_ex_enum enumerator encodes the reason why a
        /// thread is being restarted
        enum thread_state_ex_enum
        {
            wait_unknown = -1,
            wait_signaled = 0,  ///< The thread has been signaled
            wait_timeout = 1,   ///< The thread has been reactivated after a timeout
            wait_terminate = 2, ///< The thread needs to be terminated
            wait_abort = 3      ///< The thread needs to be aborted
        };

        typedef threads::detail::tagged_thread_state<thread_state_ex_enum> thread_state_ex;

        typedef thread_state_enum thread_function_type(thread_state_ex_enum);

        ///////////////////////////////////////////////////////////////////////
        /// \cond NODETAIL
        namespace detail
        {
            template <typename CoroutineImpl> struct coroutine_allocator;
        }
        /// \endcond

        typedef boost::coroutines::coroutine<
            thread_function_type, detail::coroutine_allocator> coroutine_type;
        typedef coroutine_type::thread_id_type thread_id_type;
        typedef coroutine_type::self thread_self;

        /// The function \a get_self returns a reference to the (OS thread
        /// specific) self reference to the current PX thread.
        HPX_API_EXPORT thread_self& get_self();

        /// The function \a get_self_ptr returns a pointer to the (OS thread
        /// specific) self reference to the current PX thread.
        HPX_API_EXPORT thread_self* get_self_ptr();

        /// The function \a get_ctx_ptr returns a pointer to the internal data
        /// associated with each coroutine.
        HPX_API_EXPORT thread_self::impl_type* get_ctx_ptr();

        /// The function \a get_self_ptr_checked returns a pointer to the (OS
        /// thread specific) self reference to the current PX thread.
        HPX_API_EXPORT thread_self* get_self_ptr_checked(error_code& ec = throws);

        /// The function \a get_self_id returns the PX thread id of the current
        /// thread (or zero if the current thread is not a PX thread).
        HPX_API_EXPORT thread_id_type get_self_id();

        /// The function \a get_parent_id returns the PX thread id of the
        /// currents thread parent (or zero if the current thread is not a
        /// PX thread).
        HPX_API_EXPORT thread_id_type get_parent_id();

        /// The function \a get_parent_phase returns the PX phase of the
        /// currents thread parent (or zero if the current thread is not a
        /// PX thread).
        HPX_API_EXPORT std::size_t get_parent_phase();

        /// The function \a get_parent_locality_id returns the id of the locality of
        /// the currents thread parent (or zero if the current thread is not a
        /// PX thread).
        HPX_API_EXPORT boost::uint32_t get_parent_locality_id();

        /// The function \a get_self_component_id returns the lva of the
        /// component the current thread is acting on
        HPX_API_EXPORT boost::uint64_t get_self_component_id();

        /// The function \a get_thread_manager returns a reference to the
        /// current thread manager.
        HPX_API_EXPORT threadmanager_base& get_thread_manager();

        /// The function \a get_thread_count returns the number of currently
        /// known threads.
        HPX_API_EXPORT boost::int64_t get_thread_count(
            thread_state_enum state = unknown);
    }

    /// \namespace actions
    ///
    /// The namespace \a actions contains all definitions needed for the
    /// class \a hpx#action_manager#action_manager and its related
    /// functionality. This namespace is part of the HPX core module.
    namespace actions
    {
        struct HPX_API_EXPORT base_action;
        typedef boost::shared_ptr<base_action> action_type;

        class HPX_API_EXPORT continuation;
        typedef boost::shared_ptr<continuation> continuation_type;

        class HPX_API_EXPORT action_manager;

        template <typename Component, int Action, typename Result,
            typename Arguments, typename Derived, threads::thread_priority Priority>
        struct action;
    }

    class HPX_API_EXPORT runtime;
    class HPX_API_EXPORT thread;

    /// A HPX runtime can be executed in two different modes: console mode
    /// and worker mode.
    enum runtime_mode
    {
        runtime_mode_invalid = -1,
        runtime_mode_console = 0,   ///< The runtime is the console locality
        runtime_mode_worker = 1,    ///< The runtime is a worker locality
        runtime_mode_connect = 2,   ///< The runtime is a worker locality
                                    ///< connecting late
        runtime_mode_default = 3,   ///< The runtime mode will be determined
                                    ///< based on the command line arguments
        runtime_mode_last
    };

    /// Get the readable string representing the name of the given runtime_mode
    /// constant.
    HPX_API_EXPORT char const* get_runtime_mode_name(runtime_mode state);
    HPX_API_EXPORT runtime_mode get_runtime_mode_from_name(std::string const& mode);

    ///////////////////////////////////////////////////////////////////////////
    /// Retrieve the string value of a configuration entry as given by \p key.
    HPX_API_EXPORT std::string get_config_entry(std::string const& key,
        std::string const& dflt);
    /// Retrieve the integer value of a configuration entry as given by \p key.
    HPX_API_EXPORT std::string get_config_entry(std::string const& key,
        std::size_t dflt);

    ///////////////////////////////////////////////////////////////////////////
    /// Add a function to be executed inside a HPX thread before hpx_main
    /// but guaranteed before any startup function is executed (system-wide)
    typedef HPX_STD_FUNCTION<void()> startup_function_type;
    HPX_API_EXPORT void register_pre_startup_function(startup_function_type const&);

    /// Add a function to be executed inside a HPX thread before hpx_main
    HPX_API_EXPORT void register_startup_function(startup_function_type const&);

    /// Add a function to be executed inside a HPX thread during hpx::finalize
    /// but guaranteed before any shutdown function is executed (system-wide)
    typedef HPX_STD_FUNCTION<void()> shutdown_function_type;
    HPX_API_EXPORT void register_pre_shutdown_function(shutdown_function_type const&);

    /// Add a function to be executed inside a HPX thread during hpx::finalize
    HPX_API_EXPORT void register_shutdown_function(shutdown_function_type const&);

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename SchedulingPolicy,
        typename NotificationPolicy = threads::policies::callback_notifier>
    class HPX_API_EXPORT runtime_impl;

    /// The function \a get_runtime returns a reference to the (thread
    /// specific) runtime instance.
    HPX_API_EXPORT runtime& get_runtime();
    HPX_API_EXPORT runtime* get_runtime_ptr();

    /// The function \a get_locality returns a reference to the locality
    HPX_API_EXPORT naming::locality const& get_locality();

    /// The function \a get_runtime_instance_number returns a unique number
    /// associated with the runtime instance the current thread is running in.
    HPX_API_EXPORT std::size_t get_runtime_instance_number();

    HPX_API_EXPORT void report_error(std::size_t num_thread
      , boost::exception_ptr const& e);

    HPX_API_EXPORT void report_error(boost::exception_ptr const& e);

    /// Register a function to be called during system shutdown
    HPX_API_EXPORT bool register_on_exit(HPX_STD_FUNCTION<void()>);

    enum logging_destination
    {
        destination_hpx = 0,
        destination_timing = 1,
        destination_agas = 2,
        destination_app = 3
    };

    /// \namespace components
    namespace components
    {
        enum factory_state_enum
        {
            factory_enabled  = 0,
            factory_disabled = 1,
            factory_check    = 2
        };

        /// \cond NODETAIL
        namespace detail
        {
            struct this_type {};
            struct fixed_component_tag {};
            struct simple_component_tag {};
            struct managed_component_tag {};
        }
        /// \endcond

        ///////////////////////////////////////////////////////////////////////
        typedef boost::int32_t component_type;
        enum component_enum_type
        {
            component_invalid = -1,
            component_runtime_support = 0,  // runtime support (needed to create components, etc.)
            component_memory = 1,           // general memory address
            component_memory_block = 2,     // general memory block

            // LCO's
            component_base_lco = 3,         ///< the base of all LCO's not waiting on a value
            component_base_lco_with_value = 4,
                                            ///< base LCO's blocking on a value
            component_future =              ///< a future executing the action and
                                            ///< allowing to wait for the result
                ((5 << 16) | component_base_lco_with_value),
            component_value_adaptor = 6,    ///< an adaptor to access specific slot of an LCO
            component_barrier =             ///< a LCO implementing a barrier
                ((7 << 16) | component_base_lco),
            component_thread =              ///< a ParalleX thread
                ((8 << 16) | component_base_lco_with_value),

            component_agas_primary_namespace = 12,
            component_agas_component_namespace = 13,
            component_agas_symbol_namespace = 14,

            component_last,
            component_first_dynamic = component_last,

            // force this enum type to be at least 32 bits.
            component_upper_bound = 0x7fffffffL
        };

        ///////////////////////////////////////////////////////////////////////
        template <boost::uint64_t MSB, boost::uint64_t LSB,
            typename Component = detail::this_type>
        struct fixed_component_base;

        template <typename Component>
        struct fixed_component;

        template <typename Component = detail::this_type>
        class simple_component_base;

        template <typename Component>
        class simple_component;

        template <typename Component, typename Wrapper = detail::this_type,
            typename CtorPolicy = traits::construct_without_back_ptr,
            typename DtorPolicy = traits::managed_object_controls_lifetime>
        class managed_component_base;

        template <typename Component, typename Derived = detail::this_type>
        class managed_component;

        struct HPX_API_EXPORT component_factory_base;

        template <typename Component>
        struct component_factory;

        class runtime_support;
        class memory;
        class memory_block;

        namespace stubs
        {
            struct runtime_support;
            struct memory;
            struct memory_block;
        }

        namespace server
        {
            class HPX_API_EXPORT runtime_support;
            class HPX_API_EXPORT memory;
            class HPX_API_EXPORT memory_block;
        }

        HPX_EXPORT void console_logging(logging_destination dest,
            std::size_t level, std::string const& msg);
        HPX_EXPORT void cleanup_logging();
        HPX_EXPORT void activate_logging();
    }

    HPX_EXPORT components::server::runtime_support* get_runtime_support_ptr();

    /// \namespace lcos
    namespace lcos
    {
        class base_lco;
        template <typename Result, typename RemoteResult = Result>
        class base_lco_with_value;

        template <typename Result,
            typename RemoteResult =
                typename traits::promise_remote_result<Result>::type>
        class promise;

        template <typename Action,
            typename Result = typename traits::promise_local_result<
                typename Action::result_type>::type,
            typename DirectExecute = typename Action::direct_execution>
        class packaged_action;

        template <typename Action,
            typename Result = typename traits::promise_local_result<
                typename Action::result_type>::type,
            typename DirectExecute = typename Action::direct_execution>
        class deferred_packaged_task;

        template <typename Result,
            typename RemoteResult =
                typename traits::promise_remote_result<Result>::type>
        class future;

        template <typename ValueType>
        struct object_semaphore;

        namespace stubs
        {
            template <typename ValueType>
            struct object_semaphore;
        }

        namespace server
        {
            template <typename ValueType>
            struct object_semaphore;
        }
    }

    /// \namespace util
    namespace util
    {
        class HPX_API_EXPORT section;
        class runtime_configuration;

        /// \brief Expand INI variables in a string
        HPX_API_EXPORT std::string expand(std::string const& expand);

        /// \brief Expand INI variables in a string
        HPX_API_EXPORT void expand(std::string& expand);

        template <typename Connection, typename Key = boost::uint32_t>
        class connection_cache;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Launch policy for \a hpx::async
    BOOST_SCOPED_ENUM_START(launch)
    {
        async = 0x01,
        deferred = 0x02,
        all = 0x03        // async | deferred
    };
    BOOST_SCOPED_ENUM_END

    inline bool
    operator&(BOOST_SCOPED_ENUM(launch) lhs, BOOST_SCOPED_ENUM(launch) rhs)
    {
        return static_cast<int>(lhs) & static_cast<int>(rhs) ? true : false;
    }

    /// \brief Return the list of locality ids of remote localities supporting
    ///        the given component type. By default this function will return
    ///        the list of all remote localities (all but the current locality).
    HPX_API_EXPORT std::vector<naming::id_type> find_remote_localities();
    HPX_API_EXPORT std::vector<naming::id_type> find_remote_localities(
        components::component_type);

    /// \brief Return the number of localities which are currently registered
    ///        for the running application.
    HPX_API_EXPORT boost::uint32_t get_num_localities();
    HPX_API_EXPORT boost::uint32_t get_num_localities(components::component_type);

    /// \cond NODETAIL
    namespace detail
    {
        HPX_API_EXPORT naming::gid_type get_next_id();
    }
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the number of OS-threads running in the runtime instance
    ///        the current HPX-thread is associated with.
    HPX_API_EXPORT std::size_t get_os_thread_count();

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the number of the current OS-thread running in the
    ///        runtime instance the current HPX-thread is associated with.
    ///
    /// \note   The returned value is zero based and it's maximum value is
    ///         smaller than the overall number of OS-threads executed (as
    ///         returned by \a get_os_thread_count().
    /// \note   This function needs to be executed on a HPX-thread. It will
    ///         fail otherwise (it will return -1).
    HPX_API_EXPORT std::size_t get_worker_thread_num(bool* numa_sensitive = 0);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the number of the locality this function is being called
    ///        from.
    HPX_API_EXPORT boost::uint32_t get_locality_id(error_code& ec = throws);

    /// \endcond
}

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the global id representing this locality
    ///
    /// The function \a find_here() can be used to retrieve the global id
    /// usable to refer to the current locality.
    ///
    /// \note     Generally, the id of a locality can be used for instance to
    ///           create new instances of components and to invoke plain actions
    ///           (global functions).
    ///
    /// \returns  The global id representing the locality this function has
    ///           been called on.
    ///
    /// \note     This function will return meaningful results only if called
    ///           from an HPX-thread. It will return \a hpx::naming::invalid_id
    ///           otherwise.
    ///
    /// \see      \a hpx::find_all_localities(), \a hpx::find_locality()
    HPX_API_EXPORT naming::id_type find_here();

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the list of global ids representing all localities
    ///        available to this application.
    ///
    /// The function \a find_all_localities() can be used to retrieve the
    /// global ids of all localities currently available to this application.
    ///
    /// \note     Generally, the id of a locality can be used for instance to
    ///           create new instances of components and to invoke plain actions
    ///           (global functions).
    ///
    /// \returns  The global ids representing the localities currently
    ///           available to this application.
    ///
    /// \note     This function will return meaningful results only if called
    ///           from an HPX-thread. It will return an empty vector otherwise.
    ///
    /// \see      \a hpx::find_here(), \a hpx::find_locality()
    HPX_API_EXPORT std::vector<naming::id_type> find_all_localities();

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the list of global ids representing all localities
    ///        available to this application which support the given component
    ///        type.
    ///
    /// The function \a find_all_localities() can be used to retrieve the
    /// global ids of all localities currently available to this application
    /// which support the creation of instances of the given component type.
    ///
    /// \note     Generally, the id of a locality can be used for instance to
    ///           create new instances of components and to invoke plain actions
    ///           (global functions).
    ///
    /// \param type  [in] The type of the components for which the function should
    ///           return the available localities.
    ///
    /// \returns  The global ids representing the localities currently
    ///           available to this application which support the creation of
    ///           instances of the given component type. If no localities
    ///           supporting the given component type are currently available,
    ///           this function will return an empty vector.
    ///
    /// \note     This function will return meaningful results only if called
    ///           from an HPX-thread. It will return an empty vector otherwise.
    ///
    /// \see      \a hpx::find_here(), \a hpx::find_locality()
    HPX_API_EXPORT std::vector<naming::id_type> find_all_localities(
        components::component_type type);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the global id representing an arbitrary locality which
    ///        supports the given component type.
    ///
    /// The function \a find_locality() can be used to retrieve the
    /// global ids of all localities currently available to this application
    /// which support the creation of instances of the given component type.
    ///
    /// \note     Generally, the id of a locality can be used for instance to
    ///           create new instances of components and to invoke plain actions
    ///           (global functions).
    ///
    /// \param type  [in] The type of the components for which the function should
    ///           return any available locality.
    ///
    /// \returns  The global id representing an arbitrary locality currently
    ///           available to this application which supports the creation of
    ///           instances of the given component type. If no locality
    ///           supporting the given component type is currently available,
    ///           this function will return \a hpx::naming::invalid_id.
    ///
    /// \note     This function will return meaningful results only if called
    ///           from an HPX-thread. It will return \a hpx::naming::invalid_id
    ///           otherwise.
    ///
    /// \see      \a hpx::find_here(), \a hpx::find_all_localities()
    HPX_API_EXPORT naming::id_type find_locality(components::component_type);
}

#include <hpx/lcos/async_fwd.hpp>

#endif

