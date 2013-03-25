//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_RUNTIME_JUN_10_2008_1012AM)
#define HPX_RUNTIME_RUNTIME_JUN_10_2008_1012AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/runtime/components/server/runtime_support.hpp>
#include <hpx/runtime/components/server/memory.hpp>
#include <hpx/performance_counters/registry.hpp>
#include <hpx/util/thread_mapper.hpp>
#include <hpx/util/static_reinit.hpp>
#include <hpx/util/query_counters.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    bool pre_main(runtime_mode);

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy, typename NotificationPolicy>
    class HPX_EXPORT runtime_impl;

    class HPX_EXPORT runtime
    {
    public:
        enum state
        {
            state_invalid = -1,
            state_initialized = 0,
            state_pre_startup = 1,
            state_startup = 2,
            state_pre_main = 3,
            state_running = 4,
            state_stopped = 5
        };

        state get_state() const { return state_; }

        /// The \a hpx_main_function_type is the default function type usable
        /// as the main HPX thread function.
        typedef int hpx_main_function_type();

        ///
        typedef void hpx_errorsink_function_type(
            boost::uint32_t, std::string const&);

        /// construct a new instance of a runtime
        runtime(util::runtime_configuration const& rtcfg);

        virtual ~runtime()
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
        void starting()
        {
            state_ = state_pre_main;
        }

        /// \brief Call all registered on_exit functions
        void stopping()
        {
            state_ = state_stopped;

            typedef HPX_STD_FUNCTION<void()> value_type;

            boost::mutex::scoped_lock l(on_exit_functions_mtx_);
            BOOST_FOREACH(value_type f, on_exit_functions_)
                f();
        }

        /// This accessor returns whether the runtime instance has been stopped
        bool stopped() const
        {
            return state_ == state_stopped;
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
        static std::string const* get_thread_name();

        /// \brief Return the system uptime measure on the thread executing this call
        static boost::uint64_t get_system_uptime();

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
            return *topology_;
        }

        /// \brief Install all performance counters related to this runtime
        ///        instance
        void register_counter_types();

        ///////////////////////////////////////////////////////////////////////
        virtual int run(HPX_STD_FUNCTION<hpx_main_function_type> const& func) = 0;

        virtual int run() = 0;

        virtual int start(HPX_STD_FUNCTION<hpx_main_function_type> const& func,
            bool blocking = false) = 0;

        virtual int start(bool blocking = false) = 0;

        virtual int wait() = 0;

        virtual void stop(bool blocking = true) = 0;

        virtual parcelset::parcelhandler& get_parcel_handler() = 0;

        virtual threads::threadmanager_base& get_thread_manager() = 0;

        virtual naming::resolver_client& get_agas_client() = 0;

        virtual naming::locality const& here() const = 0;

        virtual boost::uint64_t get_runtime_support_lva() const = 0;

        virtual boost::uint64_t get_memory_lva() const = 0;

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
        /// around certain library specific problems related to dynamic
        /// loading of external libraries.
        virtual bool keep_factory_alive(components::component_type type) = 0;

        /// Access one of the internal thread pools (io_service instances)
        /// HPX is using to perform specific tasks. The three possible values
        /// for the argument \p name are "main_pool", "io_pool", "parcel_pool",
        /// and "timer_pool". For any other argument value the function will
        /// return zero.
        virtual hpx::util::io_service_pool* get_thread_pool(char const* name) = 0;

        ///////////////////////////////////////////////////////////////////////
        // management API for active performance counters
        void register_query_counters(
            boost::shared_ptr<util::query_counters> active_counters)
        {
            active_counters_ = active_counters;
        }

        void start_active_counters(error_code& ec = throws);
        void stop_active_counters(error_code& ec = throws);
        void reset_active_counters(error_code& ec = throws);
        void evaluate_active_counters(bool reset = false,
            char const* description = 0, error_code& ec = throws);

        parcelset::policies::message_handler* create_message_handler(
            char const* message_handler_type, char const* action,
            parcelset::parcelport* pp, std::size_t num_messages,
            std::size_t interval, error_code& ec = throws);
        util::binary_filter* create_binary_filter(
            char const* binary_filter_type, bool compress,
            error_code& ec = throws);

    protected:
        void init_tss();
        void deinit_tss();

        friend bool hpx::pre_main(runtime_mode);
        void set_state(state s) { state_ = s; }

    protected:
        util::reinit_helper reinit_;

        // list of functions to call on exit
        typedef std::vector<HPX_STD_FUNCTION<void()> > on_exit_type;
        on_exit_type on_exit_functions_;
        boost::mutex on_exit_functions_mtx_;

        util::runtime_configuration ini_;
        boost::shared_ptr<performance_counters::registry> counters_;
        boost::shared_ptr<util::query_counters> active_counters_;

        long instance_number_;
        static boost::atomic<int> instance_number_counter_;

        // certain components (such as PAPI) require all threads to be
        // registered with the library
        util::thread_mapper thread_support_;

        boost::scoped_ptr<threads::topology> topology_;

        state state_;

        components::server::memory memory_;
        components::server::runtime_support runtime_support_;
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
