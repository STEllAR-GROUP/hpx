//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>

#include <iostream>
#include <vector>

#include <boost/config.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/foreach.hpp>

#include <hpx/state.hpp>
#include <hpx/exception.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/stringstream.hpp>
#include <hpx/runtime/components/console_error_sink.hpp>
#include <hpx/runtime/components/server/console_error_sink.hpp>
#include <hpx/runtime/components/runtime_support.hpp>
#include <hpx/runtime/parcelset/policies/global_parcelhandler_queue.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/include/performance_counters.hpp>
#include <hpx/runtime/agas/big_boot_barrier.hpp>

#include <boost/coroutine/detail/coroutine_impl_impl.hpp>
#if defined(HPX_HAVE_STACKTRACES)
#include <boost/backtrace.hpp>
#endif

#if defined(_WIN64) && defined(_DEBUG) && !defined(BOOST_COROUTINE_USE_FIBERS)
#include <io.h>
#endif

///////////////////////////////////////////////////////////////////////////////
// Make sure the system gets properly shut down while handling Ctrl-C and other
// system signals
#if defined(BOOST_WINDOWS)

namespace hpx
{
    void handle_termination(char const* reason)
    {
        std::cerr << "Received " << (reason ? reason : "unknown signal")
#if defined(HPX_HAVE_STACKTRACES)
                  << ", " << hpx::detail::backtrace()
#else
                  << "."
#endif
                  << std::endl;
        std::abort();
    }

    HPX_EXPORT BOOL WINAPI termination_handler(DWORD ctrl_type)
    {
        switch (ctrl_type) {
        case CTRL_C_EVENT:
            handle_termination("Ctrl-C");
            return TRUE;

        case CTRL_BREAK_EVENT:
            handle_termination("Ctrl-Break");
            return TRUE;

        case CTRL_CLOSE_EVENT:
            handle_termination("Ctrl-Close");
            return TRUE;

        case CTRL_LOGOFF_EVENT:
            handle_termination("Logoff");
            return TRUE;

        case CTRL_SHUTDOWN_EVENT:
            handle_termination("Shutdown");
            return TRUE;

        default:
            break;
        }
        return FALSE;
    }
}

#else

//#include <pthread.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>

namespace hpx
{
    HPX_EXPORT void termination_handler(int signum)
    {
        char* c = strsignal(signum);
        std::cerr << "Received " << (c ? c : "unknown signal")
#if defined(HPX_HAVE_STACKTRACES)
                  << ", " << hpx::detail::backtrace()
#else
                  << "."
#endif
                  << std::endl;
        std::abort();
    }
}

#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    namespace strings
    {
        char const* const runtime_mode_names[] =
        {
            "invalid",    // -1
            "console",    // 0
            "worker",     // 1
            "connect",    // 2
            "default",    // 3
        };
    }

    char const* get_runtime_mode_name(runtime_mode state)
    {
        if (state < runtime_mode_invalid || state >= runtime_mode_last)
            return "invalid (value out of bounds)";
        return strings::runtime_mode_names[state+1];
    }

    runtime_mode get_runtime_mode_from_name(std::string const& mode)
    {
        for (std::size_t i = 0; i < runtime_mode_last; ++i) {
            if (mode == strings::runtime_mode_names[i])
                return static_cast<runtime_mode>(i-1);
        }
        return runtime_mode_invalid;
    }

    ///////////////////////////////////////////////////////////////////////////
    runtime::runtime(naming::resolver_client& agas_client,
            util::runtime_configuration& rtcfg)
      : ini_(rtcfg),
        instance_number_(++instance_number_counter_),
        stopped_(true)
    {
        // initialize thread mapping for external libraries (i.e. PAPI)
        thread_support_.register_thread("main");

        // initialize our TSS
        runtime::init_tss();

        // initialize coroutines context switcher
        boost::coroutines::thread_startup("main");

        counters_.reset(new performance_counters::registry(agas_client));
    }

    ///////////////////////////////////////////////////////////////////////////
    boost::atomic<int> runtime::instance_number_counter_(-1);

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy, typename NotificationPolicy>
    runtime_impl<SchedulingPolicy, NotificationPolicy>::runtime_impl(
            util::runtime_configuration& rtcfg,
            runtime_mode locality_mode, init_scheduler_type const& init)
      : runtime(agas_client_, rtcfg),
        mode_(locality_mode), result_(0),
        io_pool_(rtcfg.get_thread_pool_size("io_pool"),
            boost::bind(&runtime_impl::init_tss, This(), "io-thread"),
            boost::bind(&runtime_impl::deinit_tss, This()), "io_pool"),
        parcel_pool_(rtcfg.get_thread_pool_size("parcel_pool"),
            boost::bind(&runtime_impl::init_tss, This(), "parcel-thread"),
            boost::bind(&runtime_impl::deinit_tss, This()), "parcel_pool"),
        timer_pool_(rtcfg.get_thread_pool_size("timer_pool"),
            boost::bind(&runtime_impl::init_tss, This(), "timer-thread"),
            boost::bind(&runtime_impl::deinit_tss, This()), "timer_pool"),
        parcel_port_(parcel_pool_, ini_.get_parcelport_address(),
            ini_.get_max_connections(), ini_.get_max_connections_per_loc()),
        agas_client_(parcel_port_, ini_, mode_),
        parcel_handler_(agas_client_, parcel_port_, &thread_manager_,
            new parcelset::policies::global_parcelhandler_queue),
        scheduler_(init),
        notifier_(boost::bind(&runtime_impl::init_tss, This(), "worker-thread"),
            boost::bind(&runtime_impl::deinit_tss, This()),
            boost::bind(&runtime_impl::report_error, This(), _1, _2)),
        thread_manager_(timer_pool_, scheduler_, notifier_),
        init_logging_(ini_, mode_ == runtime_mode_console, agas_client_),
        applier_(parcel_handler_, thread_manager_,
            boost::uint64_t(&runtime_support_), boost::uint64_t(&memory_)),
        action_manager_(applier_),
        runtime_support_(ini_, parcel_handler_.get_locality(), agas_client_, applier_)
    {
        components::server::get_error_dispatcher().register_error_sink(
            &runtime_impl::default_errorsink, default_error_sink_);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy, typename NotificationPolicy>
    runtime_impl<SchedulingPolicy, NotificationPolicy>::~runtime_impl()
    {
        LRT_(debug) << "~runtime_impl(entering)";

        // stop all services
        parcel_port_.stop();      // stops parcel_pool_ as well
        thread_manager_.stop();   // stops timer_pool_ as well
        io_pool_.stop();

        // unload libraries
        //runtime_support_.tidy();

        LRT_(debug) << "~runtime_impl(finished)";
    }

    ///////////////////////////////////////////////////////////////////////////
    bool pre_main(hpx::runtime_mode);

    template <typename SchedulingPolicy, typename NotificationPolicy>
    threads::thread_state
    runtime_impl<SchedulingPolicy, NotificationPolicy>::run_helper(
        HPX_STD_FUNCTION<runtime::hpx_main_function_type> func, int& result)
    {
        LBT_(info) << "(2nd stage) runtime_impl::run_helper: launching pre_main";

        // Change our thread description, as we're about to call pre_main
        threads::set_thread_description(threads::get_self_id(), "pre_main");

        // Finish the bootstrap
        if (!hpx::pre_main(mode_)) {
            LBT_(info) << "(3rd stage) runtime_impl::run_helper: bootstrap "
                          "aborted, bailing out";
            return threads::thread_state(threads::terminated);
        }

        LBT_(info) << "(3rd stage) runtime_impl::run_helper: bootstrap complete";

        // Now, execute the user supplied thread function (hpx_main)
        if (!!func)
        {
            // Change our thread description, as we're about to call hpx_main
            threads::set_thread_description(threads::get_self_id(), "hpx_main");

            // Call hpx_main
            result = func();
        }

        return threads::thread_state(threads::terminated);
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    int runtime_impl<SchedulingPolicy, NotificationPolicy>::start(
        HPX_STD_FUNCTION<hpx_main_function_type> func,
        std::size_t num_threads, std::size_t num_localities, bool blocking)
    {
#if defined(_WIN64) && defined(_DEBUG) && !defined(BOOST_COROUTINE_USE_FIBERS)
        // needs to be called to avoid problems at system startup
        // see: http://connect.microsoft.com/VisualStudio/feedback/ViewFeedback.aspx?FeedbackID=100319
        _isatty(0);
#endif
        // {{{ early startup code - local
        // in AGAS v2, the runtime pointer (accessible through get_runtime
        // and get_runtime_ptr) is already initialized at this point.
        applier_.init_tss();

        LRT_(info) << "cmd_line: " << get_config().get_cmd_line();

        LBT_(info) << "(1st stage) runtime_impl::start: booting locality "
                   << here() << " on " << num_threads << " OS-thread"
                   << ((num_threads == 1) ? "" : "s");

        // start runtime_support services
        runtime_support_.run();
        LBT_(info) << "(1st stage) runtime_impl::start: started "
                      "runtime_support component";

        // start the io pool
        io_pool_.run(false);
        LBT_(info) << "(1st stage) runtime_impl::start: started the application "
                      "I/O service pool";

        // start the thread manager
        thread_manager_.run(num_threads);
        LBT_(info) << "(1st stage) runtime_impl::start: started threadmanager";
        // }}}

        // invoke the AGAS v2 notifications, waking up the other localities
        agas::get_big_boot_barrier().trigger();

        // {{{ launch main
        // register the given main function with the thread manager
        LBT_(info) << "(1st stage) runtime_impl::start: launching run_helper "
                      "pxthread";

        threads::thread_init_data data(
            boost::bind(&runtime_impl::run_helper, this, func,
                boost::ref(result_)),
            "run_helper");
        thread_manager_.register_thread(data);
        this->runtime::start();
        // }}}

        // block if required
        if (blocking)
            return wait();     // wait for the shutdown_action to be executed

        return 0;   // return zero as we don't know the outcome of hpx_main yet
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    int runtime_impl<SchedulingPolicy, NotificationPolicy>::start(
        std::size_t num_threads, std::size_t num_localities, bool blocking)
    {
        HPX_STD_FUNCTION<hpx_main_function_type> empty_main;
        return start(empty_main, num_threads, num_localities, blocking);
    }

    ///////////////////////////////////////////////////////////////////////////
    static void wait_helper(components::server::runtime_support& rts,
        boost::mutex& mtx, boost::condition& cond, bool& running)
    {
        // signal successful initialization
        {
            boost::mutex::scoped_lock lk(mtx);
            running = true;
            cond.notify_all();
        }

        // wait for termination
        rts.wait();
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    int runtime_impl<SchedulingPolicy, NotificationPolicy>::wait()
    {
        LRT_(info) << "runtime_impl: about to enter wait state";

        // start the wait_helper in a separate thread
        boost::mutex mtx;
        boost::condition cond;
        bool running = false;
        boost::thread t (boost::bind(
                    &wait_helper, boost::ref(runtime_support_),
                    boost::ref(mtx), boost::ref(cond), boost::ref(running)
                )
            );

        // wait for the thread to run
        {
            boost::mutex::scoped_lock lk(mtx);
            if (!running)
                cond.wait(lk);
        }

        // block main thread
        t.join();

        LRT_(info) << "runtime_impl: exiting wait state";
        return result_;
    }

    ///////////////////////////////////////////////////////////////////////////
    // First half of termination process: stop thread manager,
    // schedule a task managed by timer_pool to initiate second part
    template <typename SchedulingPolicy, typename NotificationPolicy>
    void runtime_impl<SchedulingPolicy, NotificationPolicy>::stop(bool blocking)
    {
        LRT_(warning) << "runtime_impl: about to stop services";

        // execute all on_exit functions whenever the first thread calls this
        this->runtime::stop();

        // stop runtime_impl services (threads)
        thread_manager_.stop(false);    // just initiate shutdown

        // schedule task in timer_pool to execute stopped() below
        // this is necessary as this function (stop()) might have been called
        // from a PX thread, so it would deadlock by waiting for the thread
        // manager
        boost::mutex mtx;
        boost::condition cond;
        boost::mutex::scoped_lock l(mtx);

        boost::thread t(boost::bind(&runtime_impl::stopped, this, blocking,
            boost::ref(cond), boost::ref(mtx)));
        cond.wait(l);

        t.join();

        // stop the rest of the system
        parcel_port_.stop(blocking);        // stops parcel_pool_ as well
        io_pool_.stop();                    // stops parcel_pool_ as well

        deinit_tss();
    }

    // Second step in termination: shut down all services.
    // This gets executed as a task in the timer_pool io_service and not as
    // a PX thread!
    template <typename SchedulingPolicy, typename NotificationPolicy>
    void runtime_impl<SchedulingPolicy, NotificationPolicy>::stopped(
        bool blocking, boost::condition& cond, boost::mutex& mtx)
    {
        // wait for thread manager to exit
        runtime_support_.stopped();         // re-activate shutdown PX-thread
        thread_manager_.stop(blocking);     // wait for thread manager

        // this disables all logging from the main thread
        deinit_tss();

        LRT_(info) << "runtime_impl: stopped all services";

        boost::mutex::scoped_lock l(mtx);
        cond.notify_all();                  // we're done now
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy, typename NotificationPolicy>
    void runtime_impl<SchedulingPolicy, NotificationPolicy>::report_error(
        std::size_t num_thread, boost::exception_ptr const& e)
    {
        // Early and late exceptions, errors outside of px-threads
        if (!threads::get_self_ptr() || !threads::threadmanager_is(running))
        {
            detail::report_exception_and_terminate(e);
            return;
        }

        // The components::console_error_sink is only applied at the console,
        // so the default error sink never gets called on the locality, meaning
        // that the user never sees errors that kill the system before the
        // error parcel gets sent out. So, before we try to send the error
        // parcel (which might cause a double fault), print local diagnostics.
        components::server::console_error_sink(e);

        // Report this error to the console.
        naming::gid_type console_id;
        if (agas_client_.get_console_locality(console_id))
        {
            if (parcel_handler_.get_locality() != console_id) {
                components::console_error_sink(
                    naming::id_type(console_id, naming::id_type::unmanaged), e);
            }
        }

        components::stubs::runtime_support::terminate_all(
            naming::get_id_from_locality_id(HPX_AGAS_BOOTSTRAP_PREFIX));
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    void runtime_impl<SchedulingPolicy, NotificationPolicy>::report_error(
        boost::exception_ptr const& e)
    {
        std::size_t num_thread = hpx::threads::threadmanager_base::get_worker_thread_num();
        return report_error(num_thread, e);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy, typename NotificationPolicy>
    int runtime_impl<SchedulingPolicy, NotificationPolicy>::run(
        HPX_STD_FUNCTION<hpx_main_function_type> func,
        std::size_t num_threads, std::size_t num_localities)
    {
        // start the main thread function
        start(func, num_threads, num_localities);

        // now wait for everything to finish
        int result = wait();
        stop();

        parcel_port_.stop();      // stops parcelport for sure
        return result;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy, typename NotificationPolicy>
    int runtime_impl<SchedulingPolicy, NotificationPolicy>::run(
        std::size_t num_threads, std::size_t num_localities)
    {
        // start the main thread function
        start(num_threads, num_localities);

        // now wait for everything to finish
        int result = wait();
        stop();

        parcel_port_.stop();      // stops parcelport for sure
        return result;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy, typename NotificationPolicy>
    void runtime_impl<SchedulingPolicy, NotificationPolicy>::default_errorsink(
        std::string const& msg)
    {
        // log the exception information in any case
        LERR_(always) << "default_errorsink: unhandled exception: " << msg;

        std::cerr << msg << std::endl;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy, typename NotificationPolicy>
    void runtime_impl<SchedulingPolicy, NotificationPolicy>::init_tss(
        char const* context)
    {
        // initialize thread mapping for external libraries (i.e. PAPI)
        thread_support_.register_thread(context);

        // initialize our TSS
        this->runtime::init_tss();

        // initialize applier TSS
        applier_.init_tss();

        // initialize coroutines context switcher
        boost::coroutines::thread_startup(context);
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    void runtime_impl<SchedulingPolicy, NotificationPolicy>::deinit_tss()
    {
        // initialize coroutines context switcher
        boost::coroutines::thread_shutdown();

        // reset applier TSS
        applier_.deinit_tss();

        // reset our TSS
        this->runtime::deinit_tss();

        // reset PAPI support
        thread_support_.unregister_thread();
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    naming::gid_type
    runtime_impl<SchedulingPolicy, NotificationPolicy>::get_next_id()
    {
        return id_pool.get_id(parcel_port_.here(), agas_client_);
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    void runtime_impl<SchedulingPolicy, NotificationPolicy>::
        add_pre_startup_function(HPX_STD_FUNCTION<void()> const& f)
    {
        runtime_support_.add_pre_startup_function(f);
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    void runtime_impl<SchedulingPolicy, NotificationPolicy>::
        add_startup_function(HPX_STD_FUNCTION<void()> const& f)
    {
        runtime_support_.add_startup_function(f);
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    void runtime_impl<SchedulingPolicy, NotificationPolicy>::
        add_pre_shutdown_function(HPX_STD_FUNCTION<void()> const& f)
    {
        runtime_support_.add_pre_shutdown_function(f);
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    void runtime_impl<SchedulingPolicy, NotificationPolicy>::
        add_shutdown_function(HPX_STD_FUNCTION<void()> const& f)
    {
        runtime_support_.add_shutdown_function(f);
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    bool runtime_impl<SchedulingPolicy, NotificationPolicy>::
        keep_factory_alive(components::component_type type)
    {
        return runtime_support_.keep_factory_alive(type);
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    hpx::util::io_service_pool*
    runtime_impl<SchedulingPolicy, NotificationPolicy>::
        get_thread_pool(char const* name)
    {
        std::string service_name(name);

        if (service_name == "io_pool")
            return &io_pool_;
        if (service_name == "parcel_pool")
            return &parcel_pool_;
        if (service_name == "timer_pool")
            return &timer_pool_;

        return 0;
    }

    ///////////////////////////////////////////////////////////////////////////
    hpx::util::thread_specific_ptr<runtime *, runtime::tls_tag> runtime::runtime_;

    void runtime::init_tss()
    {
        // initialize our TSS
        BOOST_ASSERT(NULL == runtime::runtime_.get());    // shouldn't be initialized yet
        BOOST_ASSERT(NULL == threads::coroutine_type::impl_type::get_self());

        runtime::runtime_.reset(new runtime* (this));
        threads::coroutine_type::impl_type::init_self();
    }

    void runtime::deinit_tss()
    {
        // reset our TSS
        threads::coroutine_type::impl_type::reset_self();
        runtime::runtime_.reset();
    }

    /// \brief Register all performance counter types related to this runtime
    ///        instance
    void runtime::register_counter_types()
    {
        performance_counters::generic_counter_type_data statistic_counter_types[] =
        {
            // averaging counter
            { "/statistics/average", performance_counters::counter_aggregating,
              "returns the averaged value of its base counter over "
              "an arbitrary time line; pass required base counter as the instance "
              "name: /statistics{<base_counter_name>}/average",
              HPX_PERFORMANCE_COUNTER_V1,
              &performance_counters::detail::aggregating_counter_creator,
              &performance_counters::default_counter_discoverer,
              ""
            },

            // max counter
            { "/statistics/max", performance_counters::counter_aggregating,
              "returns the averaged value of its base counter over "
              "an arbitrary time line; pass required base counter as the instance "
              "name: /statistics{<base_counter_name>}/max",
              HPX_PERFORMANCE_COUNTER_V1,
              &performance_counters::detail::aggregating_counter_creator,
              &performance_counters::default_counter_discoverer,
              ""
            },

            // min counter
            { "/statistics/min", performance_counters::counter_aggregating,
              "returns the averaged value of its base counter over "
              "an arbitrary time line; pass required base counter as the instance "
              "name: /statistics{<base_counter_name>}/min",
              HPX_PERFORMANCE_COUNTER_V1,
               &performance_counters::detail::aggregating_counter_creator,
               &performance_counters::default_counter_discoverer,
              ""
            },

            // uptime counters
            { "/runtime/uptime", performance_counters::counter_elapsed_time,
              "returns the up time of the runtime instance for the referenced locality",
              HPX_PERFORMANCE_COUNTER_V1,
              &performance_counters::detail::uptime_counter_creator,
              &performance_counters::locality_counter_discoverer,
              "s"    // unit of measure is seconds
            },

            // component instance counters
            { "/runtime/component/count", performance_counters::counter_raw,
              "returns the number of component instances currently alive on "
              "this locality (the component type has to be specified as the "
              "counter parameter)",
              HPX_PERFORMANCE_COUNTER_V1,
              &performance_counters::detail::component_instance_counter_creator,
              &performance_counters::locality_counter_discoverer,
              ""
            }
        };
        performance_counters::install_counter_types(
            statistic_counter_types,
            sizeof(statistic_counter_types)/sizeof(statistic_counter_types[0]));
    }

    ///////////////////////////////////////////////////////////////////////////
    runtime& get_runtime()
    {
        BOOST_ASSERT(NULL != runtime::runtime_.get());   // should have been initialized
        return **runtime::runtime_;
    }

    runtime* get_runtime_ptr()
    {
        runtime** rt = runtime::runtime_.get();
        return rt ? *rt : NULL;
    }

    naming::locality const& get_locality()
    {
        return get_runtime().here();
    }

    void report_error(std::size_t num_thread, boost::exception_ptr const& e)
    {
        // Early and late exceptions
        if (!threads::threadmanager_is(running))
        {
            detail::report_exception_and_terminate(e);
            return;
        }

        hpx::applier::get_applier().get_thread_manager().report_error(num_thread, e);
    }

    void report_error(boost::exception_ptr const& e)
    {
        // Early and late exceptions
        if (!threads::threadmanager_is(running))
        {
            detail::report_exception_and_terminate(e);
            return;
        }

        std::size_t num_thread = hpx::threads::threadmanager_base::get_worker_thread_num();
        hpx::applier::get_applier().get_thread_manager().report_error(num_thread, e);
    }

    bool register_on_exit(HPX_STD_FUNCTION<void()> f)
    {
        runtime* rt = get_runtime_ptr();
        if (NULL == rt)
            return false;

        rt->on_exit(f);
        return true;
    }

    std::size_t get_runtime_instance_number()
    {
        runtime* rt = get_runtime_ptr();
        return (NULL == rt) ? 0 : rt->get_instance_number();
//        return get_runtime().get_instance_number();
    }

    std::string get_config_entry(std::string const& key, std::string const& dflt)
    {
        if (NULL == get_runtime_ptr())
            return "";
        return get_runtime().get_config().get_entry(key, dflt);
    }

    std::string get_config_entry(std::string const& key, std::size_t dflt)
    {
        if (NULL == get_runtime_ptr())
            return "";
        return get_runtime().get_config().get_entry(key, dflt);
    }

    ///////////////////////////////////////////////////////////////////////////
    // Helpers
    naming::id_type find_here()
    {
        if (NULL == hpx::applier::get_applier_ptr())
            return naming::invalid_id;

        return naming::id_type(hpx::applier::get_applier().get_raw_locality(),
            naming::id_type::unmanaged);
    }

    std::vector<naming::id_type>
    find_all_localities(components::component_type type)
    {
        std::vector<naming::id_type> locality_ids;
        if (NULL != hpx::applier::get_applier_ptr())
            hpx::applier::get_applier().get_localities(locality_ids, type);
        return locality_ids;
    }

    std::vector<naming::id_type> find_all_localities()
    {
        std::vector<naming::id_type> locality_ids;
        if (NULL != hpx::applier::get_applier_ptr())
            hpx::applier::get_applier().get_localities(locality_ids);
        return locality_ids;
    }

    std::vector<naming::id_type>
    find_remote_localities(components::component_type type)
    {
        std::vector<naming::id_type> locality_ids;
        if (NULL != hpx::applier::get_applier_ptr())
            hpx::applier::get_applier().get_remote_localities(locality_ids, type);
        return locality_ids;
    }

    std::vector<naming::id_type> find_remote_localities()
    {
        std::vector<naming::id_type> locality_ids;
        if (NULL != hpx::applier::get_applier_ptr())
            hpx::applier::get_applier().get_remote_localities(locality_ids);
        return locality_ids;
    }

    // find a locality supporting the given component
    naming::id_type find_locality(components::component_type type)
    {
        if (NULL == hpx::applier::get_applier_ptr())
            return naming::invalid_id;

        std::vector<naming::id_type> locality_ids;
        hpx::applier::get_applier().get_localities(locality_ids, type);

        if (locality_ids.empty()) {
            HPX_THROW_EXCEPTION(hpx::bad_component_type, "find_locality",
                "no locality supporting sheneos configuration component found");
            return naming::invalid_id;
        }

        // chose first locality to host the object
        return locality_ids.front();
    }

    /// \brief Return the number of localities which are currently registered
    ///        for the running application.
    boost::uint32_t get_num_localities()
    {
        if (NULL == hpx::applier::get_applier_ptr())
            return 0;

        // FIXME: this is overkill
        std::vector<naming::id_type> locality_ids;
        hpx::applier::get_applier().get_localities(locality_ids);
        return static_cast<boost::uint32_t>(locality_ids.size());
    }

    boost::uint32_t get_num_localities(components::component_type type)
    {
        if (NULL == hpx::applier::get_applier_ptr())
            return 0;

        // FIXME: this is overkill
        std::vector<naming::id_type> locality_ids;
        hpx::applier::get_applier().get_localities(locality_ids, type);
        return static_cast<boost::uint32_t>(locality_ids.size());
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        naming::gid_type get_next_id()
        {
            if (NULL == hpx::applier::get_applier_ptr())
                return naming::invalid_gid;

            return get_runtime().get_next_id();
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    std::size_t get_os_thread_count()
    {
        if (NULL == get_runtime_ptr())
            return 0;
        return get_runtime().get_config().get_os_thread_count();
    }

    std::size_t get_worker_thread_num(bool* numa_sensitive)
    {
        if (NULL == get_runtime_ptr())
            return std::size_t(-1);
        return get_runtime().get_thread_manager().get_worker_thread_num(numa_sensitive);
    }

    ///////////////////////////////////////////////////////////////////////////
    void register_startup_function(startup_function_type const& f)
    {
        if (NULL != get_runtime_ptr())
            get_runtime().add_startup_function(f);
    }

    void register_pre_startup_function(startup_function_type const& f)
    {
        if (NULL != get_runtime_ptr())
            get_runtime().add_pre_startup_function(f);
    }

    void register_pre_shutdown_function(shutdown_function_type const& f)
    {
        if (NULL != get_runtime_ptr())
            get_runtime().add_pre_shutdown_function(f);
    }

    void register_shutdown_function(shutdown_function_type const& f)
    {
        if (NULL != get_runtime_ptr())
            get_runtime().add_shutdown_function(f);
    }

    ///////////////////////////////////////////////////////////////////////////
    bool keep_factory_alive(components::component_type type)
    {
        if (NULL != get_runtime_ptr())
            return get_runtime().keep_factory_alive(type);
        return false;
    }

    ///////////////////////////////////////////////////////////////////////////
    components::server::runtime_support* get_runtime_support_ptr()
    {
        return reinterpret_cast<components::server::runtime_support*>(
            get_runtime().get_runtime_support_lva());
    }
}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    std::string expand(std::string const& in)
    {
        return get_runtime().get_config().expand(in);
    }

    void expand(std::string& in)
    {
        get_runtime().get_config().expand(in, std::string::size_type(-1));
    }
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming
{
    // shortcut for get_runtime().get_agas_client()
    resolver_client& get_agas_client()
    {
        return get_runtime().get_agas_client();
    }
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads
{
    // shortcut for get_applier().get_thread_manager()
    threadmanager_base& get_thread_manager()
    {
        return hpx::applier::get_applier().get_thread_manager();
    }

    // shortcut for runtime_configuration::get_default_stack_size
    std::ptrdiff_t get_default_stack_size()
    {
        return get_runtime().get_config().get_default_stack_size();
    }
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    boost::uint32_t get_locality_id(error_code& ec)
    {
        return agas::get_locality_id(ec);
    }
}

///////////////////////////////////////////////////////////////////////////////
/// explicit template instantiation for the thread manager of our choice
#if defined(HPX_GLOBAL_SCHEDULER)
#include <hpx/runtime/threads/policies/global_queue_scheduler.hpp>
template class HPX_EXPORT hpx::runtime_impl<
    hpx::threads::policies::global_queue_scheduler,
    hpx::threads::policies::callback_notifier>;
#endif

#if defined(HPX_LOCAL_SCHEDULER)
#include <hpx/runtime/threads/policies/local_queue_scheduler.hpp>
template class HPX_EXPORT hpx::runtime_impl<
    hpx::threads::policies::local_queue_scheduler,
    hpx::threads::policies::callback_notifier>;
#endif

#include <hpx/runtime/threads/policies/local_priority_queue_scheduler.hpp>
template class HPX_EXPORT hpx::runtime_impl<
    hpx::threads::policies::local_priority_queue_scheduler,
    hpx::threads::policies::callback_notifier>;

#if defined(HPX_ABP_SCHEDULER)
#include <hpx/runtime/threads/policies/abp_queue_scheduler.hpp>
template class HPX_EXPORT hpx::runtime_impl<
    hpx::threads::policies::abp_queue_scheduler,
    hpx::threads::policies::callback_notifier>;
#endif

#if defined(HPX_ABP_PRIORITY_SCHEDULER)
#include <hpx/runtime/threads/policies/abp_priority_queue_scheduler.hpp>
template class HPX_EXPORT hpx::runtime_impl<
    hpx::threads::policies::abp_priority_queue_scheduler,
    hpx::threads::policies::callback_notifier>;
#endif

#if defined(HPX_HIERARCHY_SCHEDULER)
#include <hpx/runtime/threads/policies/hierarchy_scheduler.hpp>
template class HPX_EXPORT hpx::runtime_impl<
    hpx::threads::policies::hierarchy_scheduler,
    hpx::threads::policies::callback_notifier>;
#endif

#if defined(HPX_PERIODIC_PRIORITY_SCHEDULER)
#include <hpx/runtime/threads/policies/periodic_priority_scheduler.hpp>
template class HPX_EXPORT hpx::runtime_impl<
    hpx::threads::policies::local_periodic_priority_scheduler,
    hpx::threads::policies::callback_notifier>;
#endif

