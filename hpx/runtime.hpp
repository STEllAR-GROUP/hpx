//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_RUNTIME_JUN_10_2008_1012AM)
#define HPX_RUNTIME_RUNTIME_JUN_10_2008_1012AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime_fwd.hpp>
#include <hpx/state.hpp>
#include <hpx/runtime/threads/policies/affinity_data.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/runtime/parcelset/locality.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/util/static_reinit.hpp>
#include <hpx/util/runtime_configuration.hpp>
#include <hpx/util/one_size_heap_list_base.hpp>
#include <hpx/util/thread_specific_ptr.hpp>
#if defined(HPX_HAVE_SECURITY)
#include <hpx/lcos/local/spinlock.hpp>
#endif

#if defined(HPX_HAVE_SECURITY)
#include <hpx/components/security/certificate_store.hpp>
#endif

#include <boost/smart_ptr/scoped_ptr.hpp>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    namespace util
    {
        class thread_mapper;
        class query_counters;
        class unique_id_ranges;
    }
    namespace components
    {
        struct static_factory_load_data_type;

        namespace server
        {
            class runtime_support;
            class memory;
        }
    }

    namespace performance_counters
    {
        class registry;
    }

    int pre_main(runtime_mode);

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy>
    class HPX_EXPORT runtime_impl;

#if defined(HPX_HAVE_SECURITY)
    namespace detail
    {
        struct manage_security_data;
    }
#endif

    class HPX_EXPORT runtime
    {
    public:

        state get_state() const { return state_.load(); }

        /// The \a hpx_main_function_type is the default function type usable
        /// as the main HPX thread function.
        typedef int hpx_main_function_type();

        ///
        typedef void hpx_errorsink_function_type(
            boost::uint32_t, std::string const&);

        /// construct a new instance of a runtime
        runtime(
            util::runtime_configuration & rtcfg
          , threads::policies::init_affinity_data const& affinity_init);

        virtual ~runtime();

        /// \brief Manage list of functions to call on exit
        void on_exit(util::function_nonser<void()> const& f)
        {
            std::lock_guard<boost::mutex> l(mtx_);
            on_exit_functions_.push_back(f);
        }

        /// \brief Manage runtime 'stopped' state
        void starting()
        {
            state_.store(state_pre_main);
        }

        /// \brief Call all registered on_exit functions
        void stopping()
        {
            state_.store(state_stopped);

            typedef util::function_nonser<void()> value_type;

            std::lock_guard<boost::mutex> l(mtx_);
            for (value_type const& f : on_exit_functions_)
                f();
        }

        /// This accessor returns whether the runtime instance has been stopped
        bool stopped() const
        {
            return state_.load() == state_stopped;
        }

        // the TSS holds a pointer to the runtime associated with a given
        // OS thread
        struct tls_tag {};
        static util::thread_specific_ptr<runtime*, tls_tag> runtime_;
        static util::thread_specific_ptr<std::string, tls_tag> thread_name_;
        static util::thread_specific_ptr<boost::uint64_t, tls_tag> uptime_;

        /// \brief access configuration information
        util::runtime_configuration& get_config()
        {
            return ini_;
        }
        util::runtime_configuration const& get_config() const
        {
            return ini_;
        }

        std::size_t get_instance_number() const
        {
            return static_cast<std::size_t>(instance_number_);
        }

        /// \brief Return the name of the calling thread.
        static std::string get_thread_name();

        /// \brief Return the system uptime measure on the thread executing this call
        static boost::uint64_t get_system_uptime();

        /// \brief Allow access to the registry counter registry instance used
        ///        by the HPX runtime.
        performance_counters::registry& get_counter_registry();

        /// \brief Allow access to the registry counter registry instance used
        ///        by the HPX runtime.
        performance_counters::registry const& get_counter_registry() const;

        /// \brief Return a reference to the internal PAPI thread manager
        util::thread_mapper& get_thread_mapper();

        threads::topology const& get_topology() const
        {
            return topology_;
        }

        boost::uint32_t assign_cores(std::string const& locality_basename,
            boost::uint32_t num_threads);

        boost::uint32_t assign_cores();

        /// \brief Install all performance counters related to this runtime
        ///        instance
        void register_counter_types();

        ///////////////////////////////////////////////////////////////////////
        virtual int run(util::function_nonser<hpx_main_function_type> const& func) = 0;

        virtual int run() = 0;

        virtual void rethrow_exception() = 0;

        virtual int start(util::function_nonser<hpx_main_function_type> const& func,
            bool blocking = false) = 0;

        virtual int start(bool blocking = false) = 0;

        virtual int wait() = 0;

        virtual void stop(bool blocking = true) = 0;

        virtual parcelset::parcelhandler& get_parcel_handler() = 0;
        virtual parcelset::parcelhandler const& get_parcel_handler() const = 0;

        virtual threads::threadmanager_base& get_thread_manager() = 0;

        virtual naming::resolver_client& get_agas_client() = 0;

        virtual parcelset::endpoints_type const& endpoints() const = 0;
        virtual std::string here() const = 0;

        virtual applier::applier& get_applier() = 0;

        virtual boost::uint64_t get_runtime_support_lva() const = 0;

        virtual boost::uint64_t get_memory_lva() const = 0;

        virtual void report_error(std::size_t num_thread,
            boost::exception_ptr const& e) = 0;

        virtual void report_error(boost::exception_ptr const& e) = 0;

        virtual naming::gid_type get_next_id(std::size_t count = 1) = 0;

        virtual util::unique_id_ranges& get_id_pool() = 0;

        virtual void add_pre_startup_function(util::function_nonser<void()>
            const& f) = 0;

        virtual void add_startup_function(util::function_nonser<void()> const& f) = 0;

        virtual void add_pre_shutdown_function(util::function_nonser<void()>
            const& f) = 0;

        virtual void add_shutdown_function(util::function_nonser<void()> const& f) = 0;

        /// Keep the factory object alive which is responsible for the given
        /// component type. This a purely internal function allowing to work
        /// around certain library specific problems related to dynamic
        /// loading of external libraries.
        virtual bool keep_factory_alive(components::component_type type) = 0;

        /// Access one of the internal thread pools (io_service instances)
        /// HPX is using to perform specific tasks. The three possible values
        /// for the argument \p name are "main_pool", "io_pool", "parcel_pool",
        /// and "timer_pool". For any other argument value the function will
        /// return zero.
        virtual hpx::util::io_service_pool* get_thread_pool(char const* name) = 0;

        /// \brief Register an external OS-thread with HPX
        ///
        /// This function should be called from any OS-thread which is external to
        /// HPX (not created by HPX), but which needs to access HPX functionality,
        /// such as setting a value on a promise or similar.
        ///
        /// \param name             [in] The name to use for thread registration.
        /// \param num              [in] The sequence number to use for thread
        ///                         registration. The default for this parameter
        ///                         is zero.
        /// \param service_thread   [in] The thread should be registered as a
        ///                         service thread. The default for this parameter
        ///                         is 'true'. Any service threads will be pinned
        ///                         to cores not currently used by any of the HPX
        ///                         worker threads.
        ///
        /// \note The function will compose a thread name of the form
        ///       '<name>-thread#<num>' which is used to register the thread. It
        ///       is the user's responsibility to ensure that each (composed)
        ///       thread name is unique. HPX internally uses the following names
        ///       for the threads it creates, do not reuse those:
        ///
        ///         'main', 'io', 'timer', 'parcel', 'worker'
        ///
        /// \note This function should be called for each thread exactly once. It
        ///       will fail if it is called more than once.
        ///
        /// \returns This function will return whether th erequested operation
        ///          succeeded or not.
        ///
        virtual bool register_thread(char const* name, std::size_t num = 0,
            bool service_thread = true) = 0;

        /// \brief Unregister an external OS-thread with HPX
        ///
        /// This function will unregister any external OS-thread from HPX.
        ///
        /// \note This function should be called for each thread exactly once. It
        ///       will fail if it is called more than once. It will fail as well
        ///       if the thread has not been registered before (see
        ///       \a register_thread).
        ///
        /// \returns This function will return whether th erequested operation
        ///          succeeded or not.
        ///
        virtual bool unregister_thread() = 0;

        /// Generate a new notification policy instance for the given thread
        /// name prefix
        typedef threads::policies::callback_notifier notification_policy_type;
        virtual notification_policy_type
            get_notification_policy(char const* prefix) = 0;

        /// This function creates anew base_lco_factory (if none is available
        /// for the given type yet), registers this factory with the
        /// runtime_support object and asks the factory for it's heap object
        // which can be used to create new promises of the given type.
        boost::shared_ptr<util::one_size_heap_list_base> get_promise_heap(
            components::component_type type);

        ///////////////////////////////////////////////////////////////////////
        // management API for active performance counters
        void register_query_counters(
            boost::shared_ptr<util::query_counters> const& active_counters);

        void start_active_counters(error_code& ec = throws);
        void stop_active_counters(error_code& ec = throws);
        void reset_active_counters(error_code& ec = throws);
        void evaluate_active_counters(bool reset = false,
            char const* description = 0, error_code& ec = throws);

        // stop periodic evaluation of counters during shutdown
        void stop_evaluating_counters();

        void register_message_handler(char const* message_handler_type,
            char const* action, error_code& ec = throws);
        parcelset::policies::message_handler* create_message_handler(
            char const* message_handler_type, char const* action,
            parcelset::parcelport* pp, std::size_t num_messages,
            std::size_t interval, error_code& ec = throws);
        serialization::binary_filter* create_binary_filter(
            char const* binary_filter_type, bool compress,
            serialization::binary_filter* next_filter, error_code& ec = throws);

#if defined(HPX_HAVE_SECURITY)
        components::security::signed_certificate
            get_root_certificate(error_code& ec = throws) const;
        components::security::signed_certificate
            get_certificate(error_code& ec = throws) const;

        // add a certificate for another locality
        void add_locality_certificate(
            components::security::signed_certificate const& cert);

        components::security::signed_certificate const&
            get_locality_certificate(error_code& ec) const;
        components::security::signed_certificate const&
            get_locality_certificate(boost::uint32_t locality_id, error_code& ec) const;

        void sign_parcel_suffix(
            components::security::parcel_suffix const& suffix,
            components::security::signed_parcel_suffix& signed_suffix,
            error_code& ec) const;

        template <typename Buffer>
        bool verify_parcel_suffix(Buffer const& data,
            naming::gid_type& parcel_id, error_code& ec) const;

        components::security::signed_certificate_signing_request
            get_certificate_signing_request() const;
        components::security::signed_certificate
            sign_certificate_signing_request(
                components::security::signed_certificate_signing_request csr);

        void store_root_certificate(
            components::security::signed_certificate const& root_cert);

        void store_subordinate_certificate(
            components::security::signed_certificate const& root_subca_cert,
            components::security::signed_certificate const& subca_cert);

    protected:
        void init_security();
#endif

    protected:
        void init_tss();
        void deinit_tss();

    public:
        void set_state(state s) { state_.store(s); }

    protected:
        util::reinit_helper reinit_;

        // list of functions to call on exit
        typedef std::vector<util::function_nonser<void()> > on_exit_type;
        on_exit_type on_exit_functions_;
        mutable boost::mutex mtx_;

        util::runtime_configuration ini_;
        boost::shared_ptr<performance_counters::registry> counters_;
        boost::shared_ptr<util::query_counters> active_counters_;

        long instance_number_;
        static boost::atomic<int> instance_number_counter_;

        // certain components (such as PAPI) require all threads to be
        // registered with the library
        boost::scoped_ptr<util::thread_mapper> thread_support_;

        threads::policies::init_affinity_data affinity_init_;
        threads::topology& topology_;

        // locality basename -> used cores
        typedef std::map<std::string, boost::uint32_t> used_cores_map_type;
        used_cores_map_type used_cores_map_;

        boost::atomic<state> state_;

        boost::scoped_ptr<components::server::memory> memory_;
        boost::scoped_ptr<components::server::runtime_support> runtime_support_;

#if defined(HPX_HAVE_SECURITY)
        // allocate dynamically to reduce dependencies
        mutable lcos::local::spinlock security_mtx_;
        std::unique_ptr<detail::manage_security_data> security_data_;
        components::security::certificate_store const * cert_store(error_code& ec) const;
#endif
    };

    ///////////////////////////////////////////////////////////////////////////
    /// Keep the factory object alive which is responsible for the given
    /// component type. This a purely internal function allowing to work
    /// around certain library specific problems related to dynamic
    /// loading of external libraries.
    HPX_EXPORT bool keep_factory_alive(components::component_type type);
}   // namespace hpx

#if defined(HPX_HAVE_SECURITY)
#include <hpx/components/security/verify.hpp>
namespace hpx {
    template <typename Buffer>
    bool runtime::verify_parcel_suffix(Buffer const& data,
        naming::gid_type& parcel_id, error_code& ec) const
    {
        std::lock_guard<lcos::local::spinlock> l(security_mtx_);
        return components::security::verify(*cert_store(ec), data, parcel_id);
    }
}
#endif

#include <hpx/config/warnings_suffix.hpp>

#endif
