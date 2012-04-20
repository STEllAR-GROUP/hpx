//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_RUNTIME_JUN_10_2008_1012AM)
#define HPX_RUNTIME_RUNTIME_JUN_10_2008_1012AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/io_service_pool.hpp>
#include <hpx/runtime/naming/locality.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/runtime/parcelset/parcelport.hpp>
#include <hpx/runtime/parcelset/parcelhandler.hpp>
#include <hpx/runtime/threads/policies/callback_notifier.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/actions/action_manager.hpp>
#include <hpx/runtime/components/server/runtime_support.hpp>
#include <hpx/runtime/components/server/memory.hpp>
#include <hpx/runtime/components/server/console_error_sink_singleton.hpp>
#include <hpx/performance_counters/registry.hpp>
#include <hpx/util/runtime_configuration.hpp>
#include <hpx/util/generate_unique_ids.hpp>
#include <hpx/util/thread_specific_ptr.hpp>
#include <hpx/util/thread_mapper.hpp>
#include <boost/foreach.hpp>
#include <boost/detail/atomic_count.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    template <typename SchedulingPolicy, typename NotificationPolicy>
    class HPX_EXPORT runtime_impl;

    class HPX_EXPORT runtime
    {
    public:
        /// The \a hpx_main_function_type is the default function type usable
        /// as the main HPX thread function.
        typedef int hpx_main_function_type();

        ///
        typedef void hpx_errorsink_function_type(
            boost::uint32_t, std::string const&);

        /// construct a new instance of a runtime
        runtime(naming::resolver_client& agas_client,
                util::runtime_configuration& rtcfg)
          : ini_(rtcfg),
            instance_number_(++instance_number_counter_),
            stopped_(true)
        {
            runtime::init_tss();
            counters_.reset(new performance_counters::registry(agas_client));
        }

        ~runtime()
        {
            // allow to reuse instance number if this was the only instance
            if (0 == instance_number_counter_)
                --instance_number_counter_;
        }

        /// \brief Manage list of functions to call on exit
        void on_exit(HPX_STD_FUNCTION<void()> f)
        {
            boost::mutex::scoped_lock l(on_exit_functions_mtx_);
            on_exit_functions_.push_back(f);
        }

        /// \brief Manage runtime 'stopped' state
        void start()
        {
            stopped_ = false;
        }

        /// \brief Call all registered on_exit functions
        void stop()
        {
            stopped_ = true;

            typedef HPX_STD_FUNCTION<void()> value_type;

            boost::mutex::scoped_lock l(on_exit_functions_mtx_);
            BOOST_FOREACH(value_type f, on_exit_functions_)
                f();
        }

        /// This accessor returns whether the runtime instance has been stopped
        bool stopped() const
        {
            return stopped_;
        }

        // the TSS holds a pointer to the runtime associated with a given
        // OS thread
        struct tls_tag {};
        static hpx::util::thread_specific_ptr<runtime*, tls_tag> runtime_;

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
            return (std::size_t)instance_number_;
        }

        /// \brief Allow access to the registry counter registry instance used
        ///        by the HPX runtime.
        performance_counters::registry& get_counter_registry()
        {
            return *counters_;
        }

        /// \brief Allow access to the registry counter registry instance used
        ///        by the HPX runtime.
        performance_counters::registry const& get_counter_registry() const
        {
            return *counters_;
        }

        /// \brief Return a reference to the internal PAPI thread manager
        util::thread_mapper& get_thread_mapper()
        {
            return thread_support_;
        }

        threads::topology const& get_topology() const
        {
            return topology_;
        }

        /// \brief Install all performance counters related to this runtime
        ///        instance
        void register_counter_types();

        ///////////////////////////////////////////////////////////////////////
        virtual util::io_service_pool& get_io_pool() = 0;

        virtual parcelset::parcelport& get_parcel_port() = 0;

        virtual parcelset::parcelhandler& get_parcel_handler() = 0;

        virtual threads::threadmanager_base& get_thread_manager() = 0;

        virtual naming::resolver_client& get_agas_client() = 0;

        virtual naming::locality const& here() const = 0;

        virtual std::size_t get_runtime_support_lva() const = 0;

        virtual std::size_t get_memory_lva() const = 0;

        virtual void report_error(std::size_t num_thread,
            boost::exception_ptr const& e) = 0;

        virtual void report_error(boost::exception_ptr const& e) = 0;

        virtual naming::gid_type get_next_id() = 0;

        virtual util::unique_ids& get_id_pool() = 0;

        virtual void add_pre_startup_function(HPX_STD_FUNCTION<void()> const& f) = 0;

        virtual void add_startup_function(HPX_STD_FUNCTION<void()> const& f) = 0;

        virtual void add_pre_shutdown_function(HPX_STD_FUNCTION<void()> const& f) = 0;

        virtual void add_shutdown_function(HPX_STD_FUNCTION<void()> const& f) = 0;

        /// Keep the factory object alive which is responsible for the given
        /// component type. This a purely internal function allowing to work
        /// around certain compiler specific problems related to dynamic
        /// loading of external libraries.
        virtual bool keep_factory_alive(components::component_type type) = 0;

    protected:
        void init_tss();
        void deinit_tss();

    protected:
        // list of functions to call on exit
        typedef std::vector<HPX_STD_FUNCTION<void()> > on_exit_type;
        on_exit_type on_exit_functions_;
        boost::mutex on_exit_functions_mtx_;

        util::runtime_configuration& ini_;
        boost::shared_ptr<performance_counters::registry> counters_;

        long instance_number_;
        static boost::atomic<int> instance_number_counter_;

        // certain components (such as PAPI) require all threads to be
        // registered with the library
        util::thread_mapper thread_support_;

        threads::topology topology_;

        bool stopped_;
    };

    /// The \a runtime class encapsulates the HPX runtime system in a simple to
    /// use way. It makes sure all required parts of the HPX runtime system are
    /// properly initialized.
    template <typename SchedulingPolicy, typename NotificationPolicy>
    class HPX_EXPORT runtime_impl : public runtime
    {
    private:
        typedef threads::threadmanager_impl<SchedulingPolicy, NotificationPolicy>
            threadmanager_type;

        // avoid warnings about usage of this in member initializer list
        runtime_impl* This() { return this; }

        //
        static void default_errorsink(std::string const&);

        //
        threads::thread_state run_helper(
            HPX_STD_FUNCTION<runtime::hpx_main_function_type> func, int& result);

    public:
        typedef SchedulingPolicy scheduling_policy_type;
        typedef NotificationPolicy notification_policy_type;

        typedef typename scheduling_policy_type::init_parameter_type
            init_scheduler_type;

        /// Construct a new HPX runtime instance
        ///
        /// \param locality_mode  [in] This is the mode the given runtime
        ///                       instance should be executed in.
        explicit runtime_impl(util::runtime_configuration& rtcfg,
            runtime_mode locality_mode = runtime_mode_console,
            init_scheduler_type const& init = init_scheduler_type());

        /// \brief The destructor makes sure all HPX runtime services are
        ///        properly shut down before exiting.
        ~runtime_impl();

        /// \brief Start the runtime system
        ///
        /// \param func       [in] This is the main function of an HPX
        ///                   application. It will be scheduled for execution
        ///                   by the thread manager as soon as the runtime has
        ///                   been initialized. This function is expected to
        ///                   expose an interface as defined by the typedef
        ///                   \a hpx_main_function_type.
        /// \param num_threads [in] The initial number of threads to be started
        ///                   by the threadmanager. This parameter is optional
        ///                   and defaults to 1.
        /// \param blocking   [in] This allows to control whether this
        ///                   call blocks until the runtime system has been
        ///                   stopped. If this parameter is \a true the
        ///                   function \a runtime#start will call
        ///                   \a runtime#wait internally.
        ///
        /// \returns          If a blocking is a true, this function will
        ///                   return the value as returned as the result of the
        ///                   invocation of the function object given by the
        ///                   parameter \p func. Otherwise it will return zero.
        int start(HPX_STD_FUNCTION<hpx_main_function_type> func =
                HPX_STD_FUNCTION<hpx_main_function_type>(),
            std::size_t num_threads = 1, std::size_t num_localities = 1,
            bool blocking = false);

        /// \brief Start the runtime system
        ///
        /// \param num_threads [in] The initial number of threads to be started
        ///                   by the threadmanager.
        /// \param blocking   [in] This allows to control whether this
        ///                   call blocks until the runtime system has been
        ///                   stopped. If this parameter is \a true the
        ///                   function \a runtime#start will call
        ///                   \a runtime#wait internally .
        ///
        /// \returns          If a blocking is a true, this function will
        ///                   return the value as returned as the result of the
        ///                   invocation of the function object given by the
        ///                   parameter \p func. Otherwise it will return zero.
        int start(std::size_t num_threads, std::size_t num_localities = 1,
            bool blocking = false);

        /// \brief Wait for the shutdown action to be executed
        ///
        /// \returns          This function will return the value as returned
        ///                   as the result of the invocation of the function
        ///                   object given by the parameter \p func.
        int wait();

        /// \brief Initiate termination of the runtime system
        ///
        /// \param blocking   [in] This allows to control whether this
        ///                   call blocks until the runtime system has been
        ///                   fully stopped. If this parameter is \a false then
        ///                   this call will initiate the stop action but will
        ///                   return immediately. Use a second call to stop
        ///                   with this parameter set to \a true to wait for
        ///                   all internal work to be completed.
        void stop(bool blocking = true);

        /// \brief Stop the runtime system, wait for termination
        ///
        /// \param blocking   [in] This allows to control whether this
        ///                   call blocks until the runtime system has been
        ///                   fully stopped. If this parameter is \a false then
        ///                   this call will initiate the stop action but will
        ///                   return immediately. Use a second call to stop
        ///                   with this parameter set to \a true to wait for
        ///                   all internal work to be completed.
        void stopped(bool blocking, boost::condition& cond, boost::mutex& mtx);

        /// \brief Report a non-recoverable error to the runtime system
        ///
        /// \param num_thread [in] The number of the operating system thread
        ///                   the error has been detected in.
        /// \param e          [in] This is an instance encapsulating an
        ///                   exception which lead to this function call.
        void report_error(std::size_t num_thread,
            boost::exception_ptr const& e);

        /// \brief Report a non-recoverable error to the runtime system
        ///
        /// \param e          [in] This is an instance encapsulating an
        ///                   exception which lead to this function call.
        ///
        /// \note This function will retrieve the number of the current
        ///       shepherd thread and forward to the report_error function
        ///       above.
        void report_error(boost::exception_ptr const& e);

        /// \brief Run the HPX runtime system, use the given function for the
        ///        main \a thread and block waiting for all threads to
        ///        finish
        ///
        /// \param func       [in] This is the main function of an HPX
        ///                   application. It will be scheduled for execution
        ///                   by the thread manager as soon as the runtime has
        ///                   been initialized. This function is expected to
        ///                   expose an interface as defined by the typedef
        ///                   \a hpx_main_function_type. This parameter is
        ///                   optional and defaults to none main thread
        ///                   function, in which case all threads have to be
        ///                   scheduled explicitly.
        /// \param num_threads [in] The initial number of threads to be started
        ///                   by the thread-manager. This parameter is optional
        ///                   and defaults to 1.
        /// \num_localities   [in] The overall number of localities which are
        ///                   initially used for the full application. The
        ///                   runtime system will block during startup until
        ///                   this many localities have been brought on line.
        ///                   If this is not specified the number of localities
        ///                   is assumed to be one.
        ///
        /// \note             The parameter \a func is optional. If no function
        ///                   is supplied, the runtime system will simply wait
        ///                   for the shutdown action without explicitly
        ///                   executing any main thread.
        ///
        /// \returns          This function will return the value as returned
        ///                   as the result of the invocation of the function
        ///                   object given by the parameter \p func.
        int run(HPX_STD_FUNCTION<hpx_main_function_type> func =
                    HPX_STD_FUNCTION<hpx_main_function_type>(),
                std::size_t num_threads = 1, std::size_t num_localities = 1);

        /// \brief Run the HPX runtime system, initially use the given number
        ///        of (OS) threads in the thread-manager and block waiting for
        ///        all threads to finish.
        ///
        /// \param num_threads [in] The initial number of threads to be started
        ///                   by the thread-manager.
        /// \num_localities   [in] The overall number of localities which are
        ///                   initially used for the full application. The
        ///                   runtime system will block during startup until
        ///                   this many localities have been brought on line.
        ///                   If this is not specified the number of localities
        ///                   is assumed to be one.
        ///
        /// \returns          This function will always return 0 (zero).
        int run(std::size_t num_threads, std::size_t num_localities = 1);

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename Connection>
        bool register_error_sink(F sink, Connection& conn,
            bool unregister_default = true)
        {
            if (unregister_default)
                default_error_sink_.disconnect();

            return components::server::get_error_dispatcher().
                register_error_sink(sink, conn);
        }

        ///////////////////////////////////////////////////////////////////////
        /// \brief Allow access to the AGAS client instance used by the HPX
        ///        runtime.
        naming::resolver_client& get_agas_client()
        {
            return agas_client_;
        }

        /// \brief Allow access to the parcel port instance used by the HPX
        ///        runtime.
        parcelset::parcelport& get_parcel_port()
        {
            return parcel_port_;
        }

        /// \brief Allow access to the parcel handler instance used by the HPX
        ///        runtime.
        parcelset::parcelhandler& get_parcel_handler()
        {
            return parcel_handler_;
        }

        /// \brief Allow access to the thread manager instance used by the HPX
        ///        runtime.
        threadmanager_type& get_thread_manager()
        {
            return thread_manager_;
        }

        /// \brief Allow access to the applier instance used by the HPX
        ///        runtime.
        applier::applier& get_applier()
        {
            return applier_;
        }

        /// \brief Allow access to the action manager instance used by the HPX
        ///        runtime.
        actions::action_manager& get_action_manager()
        {
            return action_manager_;
        }

        /// \brief Allow access to the locality this runtime instance is
        /// associated with.
        ///
        /// This accessor returns a reference to the locality this runtime
        /// instance is associated with.
        naming::locality const& here() const
        {
            return parcel_port_.here();
        }

        /// \brief Return the number of executed PX threads
        ///
        /// \param num This parameter specifies the sequence number of the OS
        ///            thread the number of executed PX threads should be
        ///            returned for. If this is std::size_t(-1) the function
        ///            will return the overall number of executed PX threads.
        boost::int64_t get_executed_threads(std::size_t num = std::size_t(-1)) const
        {
            return thread_manager_.get_executed_threads(num);
        }

        util::io_service_pool& get_io_pool()
        {
            return io_pool_;
        }

        std::size_t get_runtime_support_lva() const
        {
            return (std::size_t) &runtime_support_;
        }

        std::size_t get_memory_lva() const
        {
            return (std::size_t) &memory_;
        }

        naming::gid_type get_next_id();

        util::unique_ids& get_id_pool()
        {
            return id_pool;
        }

        /// Add a function to be executed inside a HPX thread before hpx_main
        /// but guaranteed to be executed before any startup function registered
        /// with \a add_startup_function.
        ///
        /// \param  f   The function 'f' will be called from inside a HPX
        ///             thread before hpx_main is executed. This is very useful
        ///             to setup the runtime environment of the application
        ///             (install performance counters, etc.)
        ///
        /// \note       The difference to a startup function is that all
        ///             pre-startup functions will be (system-wide) executed
        ///             before any startup function.
        void add_pre_startup_function(HPX_STD_FUNCTION<void()> const& f);

        /// Add a function to be executed inside a HPX thread before hpx_main
        ///
        /// \param  f   The function 'f' will be called from inside a HPX
        ///             thread before hpx_main is executed. This is very useful
        ///             to setup the runtime environment of the application
        ///             (install performance counters, etc.)
        void add_startup_function(HPX_STD_FUNCTION<void()> const& f);

        /// Add a function to be executed inside a HPX thread during
        /// hpx::finalize, but guaranteed before any of teh shutdown functions
        /// is executed.
        ///
        /// \param  f   The function 'f' will be called from inside a HPX
        ///             thread while hpx::finalize is executed. This is very
        ///             useful to tear down the runtime environment of the
        ///             application (uninstall performance counters, etc.)
        ///
        /// \note       The difference to a shutdown function is that all
        ///             pre-shutdown functions will be (system-wide) executed
        ///             before any shutdown function.
        void add_pre_shutdown_function(HPX_STD_FUNCTION<void()> const& f);

        /// Add a function to be executed inside a HPX thread during hpx::finalize
        ///
        /// \param  f   The function 'f' will be called from inside a HPX
        ///             thread while hpx::finalize is executed. This is very
        ///             useful to tear down the runtime environment of the
        ///             application (uninstall performance counters, etc.)
        void add_shutdown_function(HPX_STD_FUNCTION<void()> const& f);

        bool keep_factory_alive(components::component_type type);

    private:
        void init_tss(char const* context);
        void deinit_tss();

    private:
        util::unique_ids id_pool;
        runtime_mode mode_;
        int result_;
        util::io_service_pool io_pool_;
        util::io_service_pool parcel_pool_;
        util::io_service_pool timer_pool_;
        parcelset::parcelport parcel_port_;
        naming::resolver_client agas_client_;
        parcelset::parcelhandler parcel_handler_;
        scheduling_policy_type scheduler_;
        notification_policy_type notifier_;
        threadmanager_type thread_manager_;
        util::detail::init_logging init_logging_;
        components::server::memory memory_;
        applier::applier applier_;
        actions::action_manager action_manager_;
        components::server::runtime_support runtime_support_;
        boost::signals2::scoped_connection default_error_sink_;
    };

    ///////////////////////////////////////////////////////////////////////////
    /// Keep the factory object alive which is responsible for the given
    /// component type. This a purely internal function allowing to work
    /// around certain library specific problems related to dynamic
    /// loading of external libraries.
    HPX_EXPORT bool keep_factory_alive(components::component_type type);
}   // namespace hpx

#include <hpx/config/warnings_suffix.hpp>

#endif
