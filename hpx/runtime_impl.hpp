//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_RUNTIME_IMPL_HPP)
#define HPX_RUNTIME_RUNTIME_IMPL_HPP

#include <hpx/config.hpp>
#include <hpx/util/io_service_pool.hpp>
#include <hpx/runtime.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/runtime/parcelset/locality.hpp>
#include <hpx/runtime/parcelset/parcelport.hpp>
#include <hpx/runtime/parcelset/parcelhandler.hpp>
#include <hpx/runtime/threads/policies/affinity_data.hpp>
#include <hpx/runtime/threads/policies/callback_notifier.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/server/console_error_sink_singleton.hpp>
#include <hpx/performance_counters/registry.hpp>
#include <hpx/util/runtime_configuration.hpp>
#include <hpx/util/generate_unique_ids.hpp>
#include <hpx/util/thread_specific_ptr.hpp>
#include <hpx/util/init_logging.hpp>

#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>
#include <boost/exception_ptr.hpp>

#include <string>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx
{
    /// The \a runtime class encapsulates the HPX runtime system in a simple to
    /// use way. It makes sure all required parts of the HPX runtime system are
    /// properly initialized.
    template <typename SchedulingPolicy>
    class HPX_EXPORT runtime_impl : public runtime
    {
    private:
        // avoid warnings about usage of this in member initializer list
        runtime_impl* This() { return this; }

        //
        static void default_errorsink(std::string const&);

        //
        threads::thread_state_enum run_helper(
            util::function_nonser<runtime::hpx_main_function_type> func,
            int& result);

        void wait_helper(boost::mutex& mtx, boost::condition& cond,
            bool& running);

    public:
        typedef SchedulingPolicy scheduling_policy_type;
        typedef threads::policies::callback_notifier notification_policy_type;

        typedef typename scheduling_policy_type::init_parameter_type
            init_scheduler_type;

        /// Construct a new HPX runtime instance
        ///
        /// \param locality_mode  [in] This is the mode the given runtime
        ///                       instance should be executed in.
        /// \param num_threads    [in] The initial number of threads to be started
        ///                       by the thread-manager.
        explicit runtime_impl(util::runtime_configuration & rtcfg,
            runtime_mode locality_mode = runtime_mode_console,
            std::size_t num_threads = 1,
            init_scheduler_type const& init = init_scheduler_type(),
            threads::policies::init_affinity_data const& affinity_init =
                threads::policies::init_affinity_data());

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
        int start(util::function_nonser<hpx_main_function_type> const& func,
                bool blocking = false);

        /// \brief Start the runtime system
        ///
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
        int start(bool blocking = false);

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
        ///
        /// \note             The parameter \a func is optional. If no function
        ///                   is supplied, the runtime system will simply wait
        ///                   for the shutdown action without explicitly
        ///                   executing any main thread.
        ///
        /// \returns          This function will return the value as returned
        ///                   as the result of the invocation of the function
        ///                   object given by the parameter \p func.
        int run(util::function_nonser<hpx_main_function_type> const& func);

        /// \brief Run the HPX runtime system, initially use the given number
        ///        of (OS) threads in the thread-manager and block waiting for
        ///        all threads to finish.
        ///
        /// \returns          This function will always return 0 (zero).
        int run();

        /// Rethrow any stored exception (to be called after stop())
        void rethrow_exception();

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

        /// \brief Allow access to the parcel handler instance used by the HPX
        ///        runtime.
        parcelset::parcelhandler const& get_parcel_handler() const
        {
            return parcel_handler_;
        }

        parcelset::parcelhandler& get_parcel_handler()
        {
            return parcel_handler_;
        }

        /// \brief Allow access to the thread manager instance used by the HPX
        ///        runtime.
        hpx::threads::threadmanager_base& get_thread_manager()
        {
            return *thread_manager_;
        }

        /// \brief Allow access to the applier instance used by the HPX
        ///        runtime.
        applier::applier& get_applier()
        {
            return applier_;
        }

        /// \brief Allow access to the locality endpoints this runtime instance is
        /// associated with.
        ///
        /// This accessor returns a reference to the locality endpoints this runtime
        /// instance is associated with.
        parcelset::endpoints_type const& endpoints() const
        {
            return parcel_handler_.endpoints();
        }

        /// \brief Returns a string of the locality endpoints (usable in debug output)
        std::string here() const
        {
            std::ostringstream strm;
            strm << get_runtime().endpoints();
            return strm.str();
        }

        /// \brief Return the number of executed HPX threads
        ///
        /// \param num This parameter specifies the sequence number of the OS
        ///            thread the number of executed HPX threads should be
        ///            returned for. If this is std::size_t(-1) the function
        ///            will return the overall number of executed HPX threads.
#ifdef HPX_HAVE_THREAD_CUMULATIVE_COUNTS
        boost::int64_t get_executed_threads(std::size_t num = std::size_t(-1)) const
        {
            return thread_manager_->get_executed_threads(num);
        }
#endif

        boost::uint64_t get_runtime_support_lva() const
        {
            return reinterpret_cast<boost::uint64_t>(runtime_support_.get());
        }

        boost::uint64_t get_memory_lva() const
        {
            return reinterpret_cast<boost::uint64_t>(memory_.get());
        }

        naming::gid_type get_next_id(std::size_t count = 1);

        util::unique_id_ranges& get_id_pool()
        {
            return id_pool_;
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
        void add_pre_startup_function(util::function_nonser<void()> const& f);

        /// Add a function to be executed inside a HPX thread before hpx_main
        ///
        /// \param  f   The function 'f' will be called from inside a HPX
        ///             thread before hpx_main is executed. This is very useful
        ///             to setup the runtime environment of the application
        ///             (install performance counters, etc.)
        void add_startup_function(util::function_nonser<void()> const& f);

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
        void add_pre_shutdown_function(util::function_nonser<void()> const& f);

        /// Add a function to be executed inside a HPX thread during hpx::finalize
        ///
        /// \param  f   The function 'f' will be called from inside a HPX
        ///             thread while hpx::finalize is executed. This is very
        ///             useful to tear down the runtime environment of the
        ///             application (uninstall performance counters, etc.)
        void add_shutdown_function(util::function_nonser<void()> const& f);

        /// Keep the factory object alive which is responsible for the given
        /// component type. This a purely internal function allowing to work
        /// around certain library specific problems related to dynamic
        /// loading of external libraries.
        bool keep_factory_alive(components::component_type type);

        /// Access one of the internal thread pools (io_service instances)
        /// HPX is using to perform specific tasks. The three possible values
        /// for the argument \p name are "main_pool", "io_pool", "parcel_pool",
        /// and "timer_pool". For any other argument value the function will
        /// return zero.
        hpx::util::io_service_pool* get_thread_pool(char const* name);

        /// Register an external OS-thread with HPX
        bool register_thread(char const* name, std::size_t num = 0,
            bool service_thread = true);

        /// Unregister an external OS-thread with HPX
        bool unregister_thread();

        /// Generate a new notification policy instance for the given thread
        /// name prefix
        notification_policy_type get_notification_policy(char const* prefix);

    private:
        void init_tss(char const* context, std::size_t num, char const* postfix,
            bool service_thread);
        void deinit_tss();

    private:
        util::unique_id_ranges id_pool_;
        runtime_mode mode_;
        int result_;
        std::size_t num_threads_;
        util::io_service_pool main_pool_;
        util::io_service_pool io_pool_;
        util::io_service_pool timer_pool_;
        scheduling_policy_type scheduler_;
        notification_policy_type notifier_;
        boost::scoped_ptr<hpx::threads::threadmanager_base> thread_manager_;
        parcelset::parcelhandler parcel_handler_;
        naming::resolver_client agas_client_;
        util::detail::init_logging init_logging_;
        applier::applier applier_;
        boost::signals2::scoped_connection default_error_sink_;

        boost::mutex mtx_;
        boost::exception_ptr exception_;
    };
}

#include <hpx/config/warnings_suffix.hpp>

#endif
