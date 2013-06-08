//  Copyright (c) 2007-2013 Hartmut Kaiser
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
#include <boost/detail/scoped_enum_emulation.hpp>

#include <hpx/config.hpp>
#include <hpx/config/function.hpp>
#include <hpx/traits.hpp>
#include <hpx/lcos/local/once_fwd.hpp>
#include <hpx/util/unused.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/coroutine/coroutine.hpp>
#include <hpx/util/detail/remove_reference.hpp>
#include <hpx/runtime/threads/detail/tagged_thread_state.hpp>

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
        enum connection_type
        {
            connection_unknown = -1,
            connection_tcpip = 0,
            connection_shmem = 1,
            connection_portals4 = 2,
            connection_ibverbs = 3,
            connection_last
        };

        class HPX_API_EXPORT parcel;
        class HPX_API_EXPORT parcelport;
        class HPX_API_EXPORT parcelhandler;

        namespace server
        {
            class parcelport_queue;
        }

        struct parcelhandler_queue_base;

        namespace policies
        {
            struct global_parcelhandler_queue;
            typedef global_parcelhandler_queue parcelhandler_queue;

            struct message_handler;
        }

        HPX_API_EXPORT std::string get_connection_type_name(connection_type);
        HPX_API_EXPORT policies::message_handler* get_message_handler(
            parcelhandler* ph, char const* name, char const* type, std::size_t num,
            std::size_t interval, naming::locality const& l, connection_type t,
            error_code& ec = throws);

        HPX_API_EXPORT void flush_buffers();
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
#if defined(HPX_STATIC_PRIORITY_SCHEDULER)
            class HPX_API_EXPORT static_priority_queue_scheduler;
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
            terminated = 5,     /*!< thread has been stopped an may be
                                     garbage collected */
            staged = 6          /*!< this is not a real thread state, but
                                     allows to reference staged task descriptions,
                                     which eventually will be converted into
                                     thread objects */
        };

        /// \ cond NODETAIL
        ///   Please note that if you change the value of threads::terminated
        ///   above, you will need to adjust do_call(dummy<1> = 1) in
        ///   util/coroutine/detail/coroutine_impl.hpp as well.
        /// \ endcond

        /// \enum thread_priority
        enum thread_priority
        {
            thread_priority_unknown = -1,
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

        /// \enum thread_stacksize
        enum thread_stacksize
        {
            thread_stacksize_small = 1,         ///< use small stack size
            thread_stacksize_medium = 2,        ///< use medium sized stack size
            thread_stacksize_large = 3,         ///< use large stack size
            thread_stacksize_huge = 4,          ///< use very large stack size

            thread_stacksize_default = thread_stacksize_small,  ///< use default stack size
            thread_stacksize_minimal = thread_stacksize_small,  ///< use minimally possible stack size
            thread_stacksize_maximal = thread_stacksize_huge    ///< use maximally possible stack size
        };

        ///////////////////////////////////////////////////////////////////////
        /// \ cond NODETAIL
        namespace detail
        {
            template <typename CoroutineImpl> struct coroutine_allocator;
        }
        /// \ endcond

        typedef util::coroutines::coroutine<
            thread_function_type, detail::coroutine_allocator> coroutine_type;
        typedef coroutine_type::thread_id_type thread_id_type;
        typedef coroutine_type::self thread_self;

        ///////////////////////////////////////////////////////////////////////
        /// \ cond NODETAIL
        thread_id_type const invalid_thread_id =
            reinterpret_cast<thread_id_type>(-1);
        /// \ endcond

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
        /// thread specific) self reference to the current HPX thread.
        HPX_API_EXPORT thread_self* get_self_ptr_checked(error_code& ec = throws);

        /// The function \a get_self_id returns the HPX thread id of the current
        /// thread (or zero if the current thread is not a PX thread).
        HPX_API_EXPORT thread_id_type get_self_id();

        /// The function \a get_parent_id returns the HPX thread id of the
        /// current's thread parent (or zero if the current thread is not a
        /// PX thread).
        ///
        /// \note This function will return a meaningful value only if the
        ///       code was compiled with HPX_THREAD_MAINTAIN_PARENT_REFERENCE
        ///       being defined.
        HPX_API_EXPORT thread_id_type get_parent_id();

        /// The function \a get_parent_phase returns the HPX phase of the
        /// current's thread parent (or zero if the current thread is not a
        /// PX thread).
        ///
        /// \note This function will return a meaningful value only if the
        ///       code was compiled with HPX_THREAD_MAINTAIN_PARENT_REFERENCE
        ///       being defined.
        HPX_API_EXPORT std::size_t get_parent_phase();

        /// The function \a get_parent_locality_id returns the id of the locality of
        /// the current's thread parent (or zero if the current thread is not a
        /// PX thread).
        ///
        /// \note This function will return a meaningful value only if the
        ///       code was compiled with HPX_THREAD_MAINTAIN_PARENT_REFERENCE
        ///       being defined.
        HPX_API_EXPORT boost::uint32_t get_parent_locality_id();

        /// The function \a get_self_component_id returns the lva of the
        /// component the current thread is acting on
        ///
        /// \note This function will return a meaningful value only if the
        ///       code was compiled with HPX_THREAD_MAINTAIN_TARGET_ADDRESS
        ///       being defined.
        HPX_API_EXPORT boost::uint64_t get_self_component_id();

        /// The function \a get_thread_manager returns a reference to the
        /// current thread manager.
        HPX_API_EXPORT threadmanager_base& get_thread_manager();

        /// The function \a get_thread_count returns the number of currently
        /// known threads.
        ///
        /// \note If state == unknown this function will not only return the
        ///       number of currently existing threads, but will add the number
        ///       of registered task descriptions (which have not been
        ///       converted into threads yet).
        HPX_API_EXPORT boost::int64_t get_thread_count(
            thread_state_enum state = unknown);

        /// \copydoc get_thread_count(thread_state_enum state)
        HPX_API_EXPORT boost::int64_t get_thread_count(
            thread_priority priority, thread_state_enum state = unknown);
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

        template <typename Component, typename Result,
            typename Arguments, typename Derived>
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

        /// \ cond NODETAIL
        namespace detail
        {
            struct this_type {};
            struct fixed_component_tag {};
            struct simple_component_tag {};
            struct managed_component_tag {};
        }
        /// \ endcond

        ///////////////////////////////////////////////////////////////////////
        typedef boost::int32_t component_type;
        enum component_enum_type
        {
            component_invalid = -1,

            // Runtime support component (provides system services such as
            // component creation, etc). One per locality.
            component_runtime_support = 0,

            // Pseudo-component for direct access to local virtual memory.
            component_memory = 1,

            // Generic memory blocks.
            component_memory_block = 2,

            // Base component for LCOs that do not produce a value.
            component_base_lco = 3,

            // Base component for LCOs that do produce values.
            component_base_lco_with_value = 4,

            // Synchronization barrier LCO.
            component_barrier = ((5 << 16) | component_base_lco),

            // An LCO representing a value which may not have been computed yet.
            component_promise = ((6 << 16) | component_base_lco_with_value),

            // AGAS locality services.
            component_agas_locality_namespace = 7,

            // AGAS primary address resolution services.
            component_agas_primary_namespace = 8,

            // AGAS global type system.
            component_agas_component_namespace = 9,

            // AGAS symbolic naming services.
            component_agas_symbol_namespace = 10,

#if defined(HPX_HAVE_SODIUM)
            // root CA, subordinate CA
            component_root_certificate_authority = 11,
            component_subordinate_certificate_authority = 12,
#endif

            component_last,
            component_first_dynamic = component_last,

            // Force this enum type to be at least 32 bits.
            component_upper_bound = 0x7fffffffL //-V112
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Component = detail::this_type>
        class fixed_component_base;

        template <typename Component>
        class fixed_component;

        template <typename Component = detail::this_type>
        class abstract_simple_component_base;

        template <typename Component = detail::this_type>
        class simple_component_base;

        template <typename Component>
        class simple_component;

        template <typename Component, typename Derived = detail::this_type>
        class abstract_managed_component_base;

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

        template <typename Result>
        class future;

        template <typename Result>
        future<typename util::detail::remove_reference<Result>::type>
        make_ready_future(BOOST_FWD_REF(Result));

        future<void> make_ready_future();

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
        struct binary_filter;

        class HPX_EXPORT section;
        class HPX_EXPORT runtime_configuration;
        class HPX_EXPORT io_service_pool;

        /// \brief Expand INI variables in a string
        HPX_API_EXPORT std::string expand(std::string const& expand);

        /// \brief Expand INI variables in a string
        HPX_API_EXPORT void expand(std::string& expand);
    }

    namespace performance_counters
    {
        struct counter_info;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Launch policy for \a hpx::async
    BOOST_SCOPED_ENUM_START(launch)
    {
        async = 0x01,
        deferred = 0x02,
        task = 0x04,        // see N3632
        sync = 0x08,
        all = 0x0f          // async | deferred | task | sync
    };
    BOOST_SCOPED_ENUM_END

    inline bool
    operator&(BOOST_SCOPED_ENUM(launch) lhs, BOOST_SCOPED_ENUM(launch) rhs)
    {
        return static_cast<int>(lhs) & static_cast<int>(rhs) ? true : false;
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the number of OS-threads running in the runtime instance
    ///        the current HPX-thread is associated with.
    HPX_API_EXPORT std::size_t get_os_thread_count();

    ///////////////////////////////////////////////////////////////////////////
    HPX_API_EXPORT bool is_scheduler_numa_sensitive();

    ///////////////////////////////////////////////////////////////////////////
    HPX_API_EXPORT util::runtime_configuration const& get_config();

    ///////////////////////////////////////////////////////////////////////////
    HPX_API_EXPORT hpx::util::io_service_pool* get_thread_pool(char const* name);

    ///////////////////////////////////////////////////////////////////////////
    // Pulling important types into the main namespace
    using naming::id_type;
    using lcos::future;
    using lcos::make_ready_future;
    using lcos::promise;

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
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \note     Generally, the id of a locality can be used for instance to
    ///           create new instances of components and to invoke plain actions
    ///           (global functions).
    ///
    /// \returns  The global id representing the locality this function has
    ///           been called on.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// \note     This function will return meaningful results only if called
    ///           from an HPX-thread. It will return \a hpx::naming::invalid_id
    ///           otherwise.
    ///
    /// \see      \a hpx::find_all_localities(), \a hpx::find_locality()
    HPX_API_EXPORT naming::id_type find_here(error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the global id representing the root locality
    ///
    /// The function \a find_root_locality() can be used to retrieve the global
    /// id usable to refer to the root locality. The root locality is the
    /// locality where the main AGAS service is hosted.
    ///
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \note     Generally, the id of a locality can be used for instance to
    ///           create new instances of components and to invoke plain actions
    ///           (global functions).
    ///
    /// \returns  The global id representing the root locality for this
    ///           application.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// \note     This function will return meaningful results only if called
    ///           from an HPX-thread. It will return \a hpx::naming::invalid_id
    ///           otherwise.
    ///
    /// \see      \a hpx::find_all_localities(), \a hpx::find_locality()
    HPX_API_EXPORT naming::id_type find_root_locality(error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the list of global ids representing all localities
    ///        available to this application.
    ///
    /// The function \a find_all_localities() can be used to retrieve the
    /// global ids of all localities currently available to this application.
    ///
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \note     Generally, the id of a locality can be used for instance to
    ///           create new instances of components and to invoke plain actions
    ///           (global functions).
    ///
    /// \returns  The global ids representing the localities currently
    ///           available to this application.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// \note     This function will return meaningful results only if called
    ///           from an HPX-thread. It will return an empty vector otherwise.
    ///
    /// \see      \a hpx::find_here(), \a hpx::find_locality()
    HPX_API_EXPORT std::vector<naming::id_type> find_all_localities(
        error_code& ec = throws);

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
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \returns  The global ids representing the localities currently
    ///           available to this application which support the creation of
    ///           instances of the given component type. If no localities
    ///           supporting the given component type are currently available,
    ///           this function will return an empty vector.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// \note     This function will return meaningful results only if called
    ///           from an HPX-thread. It will return an empty vector otherwise.
    ///
    /// \see      \a hpx::find_here(), \a hpx::find_locality()
    HPX_API_EXPORT std::vector<naming::id_type> find_all_localities(
        components::component_type type, error_code& ec = throws);

    /// \brief Return the list of locality ids of remote localities supporting
    ///        the given component type. By default this function will return
    ///        the list of all remote localities (all but the current locality).
    ///
    /// The function \a find_remote_localities() can be used to retrieve the
    /// global ids of all remote localities currently available to this
    /// application (i.e. all localities except the current one).
    ///
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \note     Generally, the id of a locality can be used for instance to
    ///           create new instances of components and to invoke plain actions
    ///           (global functions).
    ///
    /// \returns  The global ids representing the remote localities currently
    ///           available to this application.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// \note     This function will return meaningful results only if called
    ///           from an HPX-thread. It will return an empty vector otherwise.
    ///
    /// \see      \a hpx::find_here(), \a hpx::find_locality()
    HPX_API_EXPORT std::vector<naming::id_type> find_remote_localities(
        error_code& ec = throws);

    /// \brief Return the list of locality ids of remote localities supporting
    ///        the given component type. By default this function will return
    ///        the list of all remote localities (all but the current locality).
    ///
    /// The function \a find_remote_localities() can be used to retrieve the
    /// global ids of all remote localities currently available to this
    /// application (i.e. all localities except the current one) which
    /// support the creation of instances of the given component type.
    ///
    /// \param type  [in] The type of the components for which the function should
    ///           return the available remote localities.
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \note     Generally, the id of a locality can be used for instance to
    ///           create new instances of components and to invoke plain actions
    ///           (global functions).
    ///
    /// \returns  The global ids representing the remote localities currently
    ///           available to this application.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// \note     This function will return meaningful results only if called
    ///           from an HPX-thread. It will return an empty vector otherwise.
    ///
    /// \see      \a hpx::find_here(), \a hpx::find_locality()
    HPX_API_EXPORT std::vector<naming::id_type> find_remote_localities(
        components::component_type type, error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the global id representing an arbitrary locality which
    ///        supports the given component type.
    ///
    /// The function \a find_locality() can be used to retrieve the
    /// global id of an arbitrary locality currently available to this
    /// application which supports the creation of instances of the given
    /// component type.
    ///
    /// \note     Generally, the id of a locality can be used for instance to
    ///           create new instances of components and to invoke plain actions
    ///           (global functions).
    ///
    /// \param type  [in] The type of the components for which the function should
    ///           return any available locality.
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \returns  The global id representing an arbitrary locality currently
    ///           available to this application which supports the creation of
    ///           instances of the given component type. If no locality
    ///           supporting the given component type is currently available,
    ///           this function will return \a hpx::naming::invalid_id.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// \note     This function will return meaningful results only if called
    ///           from an HPX-thread. It will return \a hpx::naming::invalid_id
    ///           otherwise.
    ///
    /// \see      \a hpx::find_here(), \a hpx::find_all_localities()
    HPX_API_EXPORT naming::id_type find_locality(components::component_type type,
        error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the number of localities which are currently registered
    ///        for the running application.
    ///
    /// The function \a get_num_localities returns the number of localities
    /// currently connected to the console.
    ///
    /// \note     This function will return meaningful results only if called
    ///           from an HPX-thread. It will return 0 otherwise.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// \see      \a hpx::find_all_localities, \a hpx::get_num_localities_async
    HPX_API_EXPORT boost::uint32_t get_num_localities(error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Asynchronously return the number of localities which are
    ///        currently registered for the running application.
    ///
    /// The function \a get_num_localities_async asynchronously returns the
    /// number of localities currently connected to the console. The returned
    /// future represents the actual result.
    ///
    /// \note     This function will return meaningful results only if called
    ///           from an HPX-thread. It will return 0 otherwise.
    ///
    /// \see      \a hpx::find_all_localities, \a hpx::get_num_localities
    HPX_API_EXPORT lcos::future<boost::uint32_t> get_num_localities_async();

    /// \brief Return the number of localities which are currently registered
    ///        for the running application.
    ///
    /// The function \a get_num_localities returns the number of localities
    /// currently connected to the console which support the creation of the
    /// given component type.
    ///
    /// \param t  The component type for which the number of connected
    ///           localities should be retrieved.
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \note     This function will return meaningful results only if called
    ///           from an HPX-thread. It will return 0 otherwise.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// \see      \a hpx::find_all_localities, \a hpx::get_num_localities_async
    HPX_API_EXPORT boost::uint32_t get_num_localities(
        components::component_type t, error_code& ec = throws);

    /// \brief Asynchronously return the number of localities which are
    ///        currently registered for the running application.
    ///
    /// The function \a get_num_localities_async asynchronously returns the
    /// number of localities currently connected to the console which support
    /// the creation of the given component type. The returned future represents
    /// the actual result.
    ///
    /// \param t  The component type for which the number of connected
    ///           localities should be retrieved.
    ///
    /// \note     This function will return meaningful results only if called
    ///           from an HPX-thread. It will return 0 otherwise.
    ///
    /// \see      \a hpx::find_all_localities, \a hpx::get_num_localities
    HPX_API_EXPORT lcos::future<boost::uint32_t> get_num_localities_async(
        components::component_type t);

    ///////////////////////////////////////////////////////////////////////////
    /// The type of a function which is registered to be executed as a
    /// startup or pre-startup function.
    typedef HPX_STD_FUNCTION<void()> startup_function_type;

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Add a function to be executed by a HPX thread before hpx_main
    /// but guaranteed before any startup function is executed (system-wide).
    ///
    /// Any of the functions registered with \a register_pre_startup_function
    /// are guaranteed to be executed by an HPX thread before any of the
    /// registered startup functions are executed (see
    /// \a hpx::register_startup_function()).
    ///
    /// \param f  [in] The function to be registered to run by an HPX thread as
    ///           a pre-startup function.
    ///
    /// \note If this function is called while the pre-startup functions are
    ///       being executed or after that point, it will raise a invalid_status
    ///       exception.
    ///
    ///       This function is one of the few API functions which can be called
    ///       before the runtime system has been fully initialized. It will
    ///       automatically stage the provided startup function to the runtime
    ///       system during its initialization (if necessary).
    ///
    /// \see    \a hpx::register_startup_function()
    HPX_API_EXPORT void register_pre_startup_function(startup_function_type const& f);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Add a function to be executed by a HPX thread before hpx_main
    /// but guaranteed after any pre-startup function is executed (system-wide).
    ///
    /// Any of the functions registered with \a register_startup_function
    /// are guaranteed to be executed by an HPX thread after any of the
    /// registered pre-startup functions are executed (see:
    /// \a hpx::register_pre_startup_function()), but before \a hpx_main is
    /// being called.
    ///
    /// \param f  [in] The function to be registered to run by an HPX thread as
    ///           a startup function.
    ///
    /// \note If this function is called while the startup functions are
    ///       being executed or after that point, it will raise a invalid_status
    ///       exception.
    ///
    ///       This function is one of the few API functions which can be called
    ///       before the runtime system has been fully initialized. It will
    ///       automatically stage the provided startup function to the runtime
    ///       system during its initialization (if necessary).
    ///
    /// \see    \a hpx::register_pre_startup_function()
    HPX_API_EXPORT void register_startup_function(startup_function_type const& f);

    /// The type of a function which is registered to be executed as a
    /// shutdown or pre-shutdown function.
    typedef HPX_STD_FUNCTION<void()> shutdown_function_type;

    /// \brief Add a function to be executed by a HPX thread during
    /// \a hpx::finalize() but guaranteed before any shutdown function is
    /// executed (system-wide)
    ///
    /// Any of the functions registered with \a register_pre_shutdown_function
    /// are guaranteed to be executed by an HPX thread during the execution of
    /// \a hpx::finalize() before any of the registered shutdown functions are
    /// executed (see: \a hpx::register_shutdown_function()).
    ///
    /// \param f  [in] The function to be registered to run by an HPX thread as
    ///           a pre-shutdown function.
    ///
    /// \note If this function is called before the runtime system is
    ///       initialized, or while the pre-shutdown functions are
    ///       being executed, or after that point, it will raise a invalid_status
    ///       exception.
    ///
    /// \see    \a hpx::register_shutdown_function()
    HPX_API_EXPORT void register_pre_shutdown_function(shutdown_function_type const& f);

    /// \brief Add a function to be executed by a HPX thread during
    /// \a hpx::finalize() but guaranteed after any pre-shutdown function is
    /// executed (system-wide)
    ///
    /// Any of the functions registered with \a register_shutdown_function
    /// are guaranteed to be executed by an HPX thread during the execution of
    /// \a hpx::finalize() after any of the registered pre-shutdown functions
    /// are executed (see: \a hpx::register_pre_shutdown_function()).
    ///
    /// \param f  [in] The function to be registered to run by an HPX thread as
    ///           a shutdown function.
    ///
    /// \note If this function is called before the runtime system is
    ///       initialized, or while the shutdown functions are
    ///       being executed, or after that point, it will raise a invalid_status
    ///       exception.
    ///
    /// \see    \a hpx::register_pre_shutdown_function()
    HPX_API_EXPORT void register_shutdown_function(shutdown_function_type const& f);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the number of the current OS-thread running in the
    ///        runtime instance the current HPX-thread is executed with.
    ///
    /// This function returns the zero based index of the OS-thread which
    /// executes the current HPX-thread.
    ///
    /// \note   The returned value is zero based and it's maximum value is
    ///         smaller than the overall number of OS-threads executed (as
    ///         returned by \a get_os_thread_count().
    ///
    /// \note   This function needs to be executed on a HPX-thread. It will
    ///         fail otherwise (it will return -1).
    HPX_API_EXPORT std::size_t get_worker_thread_num();

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the number of the locality this function is being called
    ///        from.
    ///
    /// This function returns the id of the current locality.
    ///
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \note     The returned value is zero based and it's maximum value is
    ///           smaller than the overall number of localities the current
    ///           application is running on (as returned by
    ///           \a get_num_localities()).
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// \note     This function needs to be executed on a HPX-thread. It will
    ///           fail otherwise (it will return -1).
    HPX_API_EXPORT boost::uint32_t get_locality_id(error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Test whether the runtime system is currently running.
    ///
    /// This function returns whether the runtime system is currently running
    /// or not, e.g.  whether the current state of the runtime system is
    /// \a hpx::runtime::running
    ///
    /// \note   This function needs to be executed on a HPX-thread. It will
    ///         return false otherwise.
    HPX_API_EXPORT bool is_running();

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the name of the calling thread.
    ///
    /// This function returns the name of the calling thread. This name uniquely
    /// identifies the thread in the context of HPX. If the function is called
    /// while no HPX runtime system is active, it will return zero.
    HPX_API_EXPORT std::string const* get_thread_name();

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the number of worker OS- threads used to execute HPX
    ///        threads
    ///
    /// This function returns the number of OS-threads used to execute HPX
    /// threads. If the function is called while no HPX runtime system is active,
    /// it will return zero.
    HPX_API_EXPORT std::size_t get_num_worker_threads();

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the system uptime measure on the thread executing this call.
    ///
    /// This function returns the system uptime measured in nanoseconds for the
    /// thread executing this call. If the function is called while no HPX
    /// runtime system is active, it will return zero.
    HPX_API_EXPORT boost::uint64_t get_system_uptime();

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the id of the locality where the object referenced by the
    ///        given id is currently located on
    ///
    /// The function hpx::get_colocation_id() returns the id of the locality
    /// where the given object is currently located.
    ///
    /// \param id [in] The id of the object to locate.
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// \see    \a hpx::get_colocation_id_async()
    HPX_API_EXPORT naming::id_type get_colocation_id(naming::id_type id,
        error_code& ec = throws);

    /// \brief Asynchronously return the id of the locality where the object 
    ///        referenced by the given id is currently located on
    ///
    /// \see    \a hpx::get_colocation_id()
    HPX_API_EXPORT lcos::future<naming::id_type> get_colocation_id_async(
        naming::id_type id);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Trigger the LCO referenced by the given id
    ///
    /// \param id [in] this represents the id of the LCO which should be
    ///           triggered.
    HPX_API_EXPORT void trigger_lco_event(naming::id_type const& id);

    /// \brief Set the result value for the LCO referenced by the given id
    ///
    /// \param id [in] this represents the id of the LCO which should
    ///           receive the given value.
    /// \param t  [in] this is the value which should be sent to the LCO.
    template <typename T>
    void set_lco_value(naming::id_type const& id, BOOST_FWD_REF(T) t);

    /// \brief Set the error state for the LCO referenced by the given id
    ///
    /// \param id [in] this represents the id of the LCO which should
    ///           eceive the error value.
    /// \param e  [in] this is the error value which should be sent to
    ///           the LCO.
    HPX_API_EXPORT void set_lco_error(naming::id_type const& id,
        boost::exception_ptr const& e);

    /// \copydoc hpx::set_lco_error(naming::id_type const& id, boost::exception_ptr const& e)
    HPX_API_EXPORT void set_lco_error(naming::id_type const& id,
        BOOST_RV_REF(boost::exception_ptr) e);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Start all active performance counters, optionally naming the
    ///        section of code
    ///
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// \note     The active counters are those which have been specified on
    ///           the command line while executing the application (see command
    ///           line option \--hpx:print-counter)
    HPX_API_EXPORT void start_active_counters(error_code& ec = throws);

    /// \brief Resets all active performance counters.
    ///
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// \note     The active counters are those which have been specified on
    ///           the command line while executing the application (see command
    ///           line option \--hpx:print-counter)
    HPX_API_EXPORT void reset_active_counters(error_code& ec = throws);

    /// \brief Stop all active performance counters.
    ///
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// \note     The active counters are those which have been specified on
    ///           the command line while executing the application (see command
    ///           line option \--hpx:print-counter)
    HPX_API_EXPORT void stop_active_counters(error_code& ec = throws);

    /// \brief Evaluate and output all active performance counters, optionally
    ///        naming the point in code marked by this function.
    ///
    /// \param reset       [in] this is an optional flag allowing to reset
    ///                    the counter value after it has been evaluated.
    /// \param description [in] this is an optional value naming the point in
    ///                    the code marked by the call to this function.
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// \note     The output generated by this function is redirected to the
    ///           destination specified by the corresponding command line
    ///           options (see \--hpx:print-counter-destination).
    ///
    /// \note     The active counters are those which have been specified on
    ///           the command line while executing the application (see command
    ///           line option \--hpx:print-counter)
    HPX_API_EXPORT void evaluate_active_counters(bool reset = false,
        char const* description = 0, error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Create an instance of a message handler plugin
    ///
    /// The function hpx::create_message_handler() creates an instance of a
    /// message handler plugin based on the parameters specified.
    ///
    /// \param message_handler_type
    /// \param action
    /// \param pp
    /// \param num_messages
    /// \param interval
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    HPX_API_EXPORT parcelset::policies::message_handler* create_message_handler(
        char const* message_handler_type, char const* action,
        parcelset::parcelport* pp, std::size_t num_messages,
        std::size_t interval, error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Create an instance of a binary filter plugin
    ///
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    HPX_API_EXPORT util::binary_filter* create_binary_filter(
        char const* binary_filter_type, bool compress, 
        util::binary_filter* next_filter = 0, error_code& ec = throws);

#if defined(HPX_HAVE_SODIUM)
    namespace components { namespace security
    {
        class certificate;
        class parcel_suffix;
        class hash;

        template <typename T> class signed_type;
        typedef signed_type<certificate> signed_certificate;
        typedef signed_type<parcel_suffix> signed_parcel_suffix;
    }}

#if defined(HPX_HAVE_SECURITY)
    /// \brief Return the certificate for this locality
    ///
    /// \returns This function returns the signed certificate for this locality.
    HPX_API_EXPORT components::security::signed_certificate const&
        get_locality_certificate(error_code& ec = throws);

    /// \brief Return the certificate for the given locality
    ///
    /// \param id The id representing the locality for which to retrieve
    ///           the signed certificate.
    ///
    /// \returns This function returns the signed certificate for the locality
    ///          identified by the parameter \a id.
    HPX_API_EXPORT components::security::signed_certificate const&
        get_locality_certificate(naming::id_type const& id, error_code& ec = throws);

    /// \brief Add the given certificate to the certificate store of this locality.
    ///
    /// \param cert The certificate to add to the certificate store of this
    ///             locality
    HPX_API_EXPORT void add_locality_certificate(
        components::security::signed_certificate const& cert,
        error_code& ec = throws);

    /// \brief Sign the given parcel-suffix
    ///
    /// \param suffix         The parcel suffoix to be signed
    /// \param signed_suffix  The signed parcel suffix will be placed here
    ///
    HPX_API_EXPORT void sign_parcel_suffix(
        components::security::parcel_suffix const& suffix,
        components::security::signed_parcel_suffix& signed_suffix,
        error_code& ec = throws);

    /// \brief Verify the certificate in the given byte sequence
    ///
    /// \param data      The full received message buffer, assuming that it
    ///                  has a parcel_suffix appended.
    /// \param parcel_id The parcel id of the first parcel in side the message
    ///
    HPX_API_EXPORT bool verify_parcel_suffix(std::vector<char> const& data,
        naming::gid_type& parcel_id, error_code& ec = throws);

#endif
#endif
}

#include <hpx/lcos/async_fwd.hpp>

#endif

