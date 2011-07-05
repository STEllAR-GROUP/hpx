//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <vector>

#include <hpx/hpx_fwd.hpp>

#include <boost/config.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/foreach.hpp>
#include <boost/io/ios_state.hpp>

#include <hpx/include/runtime.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/stringstream.hpp>
#include <hpx/runtime/components/console_error_sink.hpp>
#include <hpx/runtime/components/runtime_support.hpp>
#include <hpx/runtime/parcelset/policies/global_parcelhandler_queue.hpp>

#if HPX_AGAS_VERSION > 0x10
    #include <hpx/runtime/agas/router/big_boot_barrier.hpp>
#endif
 
///////////////////////////////////////////////////////////////////////////////
// Make sure the system gets properly shut down while handling Ctrl-C and other
// system signals
#if defined(BOOST_WINDOWS)

static boost::function0<void> console_ctrl_function;

BOOL WINAPI console_ctrl_handler(DWORD ctrl_type)
{
    switch (ctrl_type) {
    case CTRL_C_EVENT:
    case CTRL_BREAK_EVENT:
    case CTRL_CLOSE_EVENT:
    case CTRL_SHUTDOWN_EVENT:
        console_ctrl_function();
        return TRUE;
        
    default:
        return FALSE;
    }
}

#else

#include <iostream>
#include <pthread.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#if defined(HPX_STACKTRACES)
    #include <boost/backtrace.hpp>
#endif

void hpx_termination_handler(int signum)
{
    char* c = strsignal(signum); 
    std::cerr << "Received " << (c ? c : "unknown signal")
    #if defined(HPX_STACKTRACES)
              << ", " << boost::trace() 
    #else
              << "."
    #endif
              << std::endl;
    ::abort();
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
            "invalid",
            "console",
            "worker",
        };
    }

    char const* get_runtime_mode_name(runtime_mode state)
    {
        if (state < runtime_mode_invalid || state > runtime_mode_worker)
            return "invalid";
        return strings::runtime_mode_names[state];
    }

    ///////////////////////////////////////////////////////////////////////////
    boost::atomic<int> runtime::instance_number_counter_(-1);

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy, typename NotificationPolicy> 
    runtime_impl<SchedulingPolicy, NotificationPolicy>::runtime_impl(
            std::string const& address, boost::uint16_t port,
            std::string const& agas_address, boost::uint16_t agas_port, 
            runtime_mode locality_mode, init_scheduler_type const& init,
            std::string const& hpx_ini_file,
            std::vector<std::string> const& cmdline_ini_defs) 
      : runtime(agas_client_, hpx_ini_file, cmdline_ini_defs),
        mode_(locality_mode), result_(0), 
#if HPX_AGAS_VERSION <= 0x10
        agas_pool_(boost::bind(&runtime_impl::init_tss, This()),
            boost::bind(&runtime_impl::deinit_tss, This()), "agas_client_pool"), 
#else
        io_pool_(boost::bind(&runtime_impl::init_tss, This()),
            boost::bind(&runtime_impl::deinit_tss, This()), "io_pool"), 
#endif
        parcel_pool_(boost::bind(&runtime_impl::init_tss, This()),
            boost::bind(&runtime_impl::deinit_tss, This()), "parcel_pool"), 
        timer_pool_(boost::bind(&runtime_impl::init_tss, This()),
            boost::bind(&runtime_impl::deinit_tss, This()), "timer_pool"),
        parcel_port_(parcel_pool_, naming::locality(address, port)),
#if HPX_AGAS_VERSION <= 0x10
        agas_client_(agas_pool_, naming::locality(agas_address, agas_port),
                     ini_, mode_ == runtime_mode_console),
#else
        agas_client_(parcel_port_, ini_, mode_),
#endif
        parcel_handler_(agas_client_, parcel_port_, &thread_manager_
                       , new parcelset::policies::global_parcelhandler_queue),
#if HPX_AGAS_VERSION <= 0x10
        init_logging_(ini_, mode_ == runtime_mode_console, agas_client_),
#endif
        scheduler_(init),
        notifier_(boost::bind(&runtime_impl::init_tss, This()),
            boost::bind(&runtime_impl::deinit_tss, This()),
            boost::bind(&runtime_impl::report_error, This(), _1, _2)),
        thread_manager_(timer_pool_, scheduler_, notifier_),
#if HPX_AGAS_VERSION > 0x10
        init_logging_(ini_, mode_ == runtime_mode_console, agas_client_),
#endif
        applier_(parcel_handler_, thread_manager_, 
            boost::uint64_t(&runtime_support_), boost::uint64_t(&memory_)),
        action_manager_(applier_),
        runtime_support_(ini_, parcel_handler_.get_prefix(), agas_client_, applier_)
    {
        components::server::get_error_dispatcher().register_error_sink(
            &runtime_impl::default_errorsink, default_error_sink_);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy, typename NotificationPolicy> 
    runtime_impl<SchedulingPolicy, NotificationPolicy>::runtime_impl(
            naming::locality address, naming::locality agas_address, 
            runtime_mode locality_mode, init_scheduler_type const& init,
            std::string const& hpx_ini_file,
            std::vector<std::string> const& cmdline_ini_defs) 
      : runtime(agas_client_, hpx_ini_file, cmdline_ini_defs),
        mode_(locality_mode), result_(0), 
#if HPX_AGAS_VERSION <= 0x10
        agas_pool_(boost::bind(&runtime_impl::init_tss, This()),
            boost::bind(&runtime_impl::deinit_tss, This()), "agas_client_pool"), 
#else
        io_pool_(boost::bind(&runtime_impl::init_tss, This()),
            boost::bind(&runtime_impl::deinit_tss, This()), "io_pool"), 
#endif
        parcel_pool_(boost::bind(&runtime_impl::init_tss, This()),
            boost::bind(&runtime_impl::deinit_tss, This()), "parcel_pool"), 
        timer_pool_(boost::bind(&runtime_impl::init_tss, This()),
            boost::bind(&runtime_impl::deinit_tss, This()), "timer_pool"),
        parcel_port_(parcel_pool_, address),
#if HPX_AGAS_VERSION <= 0x10
        agas_client_(agas_pool_, agas_address, ini_, mode_ == runtime_mode_console),
#else
        agas_client_(parcel_port_, ini_, mode_),
#endif
        parcel_handler_(agas_client_, parcel_port_, &thread_manager_
                       , new parcelset::policies::global_parcelhandler_queue),
#if HPX_AGAS_VERSION <= 0x10
        init_logging_(ini_, mode_ == runtime_mode_console, agas_client_),
#endif
        scheduler_(init),
        notifier_(boost::bind(&runtime_impl::init_tss, This()),
            boost::bind(&runtime_impl::deinit_tss, This()),
            boost::bind(&runtime_impl::report_error, This(), _1, _2)),
        thread_manager_(timer_pool_, scheduler_, notifier_),
#if HPX_AGAS_VERSION > 0x10
        init_logging_(ini_, mode_ == runtime_mode_console, agas_client_),
#endif
        applier_(parcel_handler_, thread_manager_, 
            boost::uint64_t(&runtime_support_), boost::uint64_t(&memory_)),
        action_manager_(applier_),
        runtime_support_(ini_, parcel_handler_.get_prefix(), agas_client_, applier_)
    {
        components::server::get_error_dispatcher().register_error_sink(
            &runtime_impl::default_errorsink, default_error_sink_);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy, typename NotificationPolicy> 
    runtime_impl<SchedulingPolicy, NotificationPolicy>::runtime_impl(
            naming::locality address, runtime_mode locality_mode, 
            init_scheduler_type const& init,
            std::string const& hpx_ini_file,
            std::vector<std::string> const& cmdline_ini_defs) 
      : runtime(agas_client_, hpx_ini_file, cmdline_ini_defs),
        mode_(locality_mode), result_(0), 
#if HPX_AGAS_VERSION <= 0x10
        agas_pool_(boost::bind(&runtime_impl::init_tss, This()),
            boost::bind(&runtime_impl::deinit_tss, This()), "agas_client_pool"), 
#else
        io_pool_(boost::bind(&runtime_impl::init_tss, This()),
            boost::bind(&runtime_impl::deinit_tss, This()), "io_pool"), 
#endif
        parcel_pool_(boost::bind(&runtime_impl::init_tss, This()),
            boost::bind(&runtime_impl::deinit_tss, This()), "parcel_pool"), 
        timer_pool_(boost::bind(&runtime_impl::init_tss, This()),
            boost::bind(&runtime_impl::deinit_tss, This()), "timer_pool"),
        parcel_port_(parcel_pool_, address),
#if HPX_AGAS_VERSION <= 0x10
        agas_client_(agas_pool_, ini_, mode_ == runtime_mode_console),
#else
        agas_client_(parcel_port_, ini_, mode_),
#endif
        parcel_handler_(agas_client_, parcel_port_, &thread_manager_
                       , new parcelset::policies::global_parcelhandler_queue),
#if HPX_AGAS_VERSION <= 0x10
        init_logging_(ini_, mode_ == runtime_mode_console, agas_client_),
#endif
        scheduler_(init),
        notifier_(boost::bind(&runtime_impl::init_tss, This()),
            boost::bind(&runtime_impl::deinit_tss, This()),
            boost::bind(&runtime_impl::report_error, This(), _1, _2)),
        thread_manager_(timer_pool_, scheduler_, notifier_),
#if HPX_AGAS_VERSION > 0x10
        init_logging_(ini_, mode_ == runtime_mode_console, agas_client_),
#endif
        applier_(parcel_handler_, thread_manager_, 
            boost::uint64_t(&runtime_support_), boost::uint64_t(&memory_)),
        action_manager_(applier_),
        runtime_support_(ini_, parcel_handler_.get_prefix(), agas_client_, applier_)
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
#if HPX_AGAS_VERSION <= 0x10
        agas_pool_.stop();
#else
        io_pool_.stop();
#endif
        // unload libraries
        //runtime_support_.tidy();

        LRT_(debug) << "~runtime_impl(finished)";
    }

    ///////////////////////////////////////////////////////////////////////////
#if defined(_WIN64) && defined(_DEBUG) && !defined(BOOST_COROUTINE_USE_FIBERS)
#include <io.h>
#endif

    ///////////////////////////////////////////////////////////////////////////
#if HPX_AGAS_VERSION <= 0x10
    template <typename SchedulingPolicy, typename NotificationPolicy> 
    threads::thread_state 
    runtime_impl<SchedulingPolicy, NotificationPolicy>::run_helper(
        boost::function<runtime::hpx_main_function_type> func, int& result)
    {
        // if we're not the console, we'll pull the console configuration 
        // information and merge it with ours
//         if (mode_ == worker) {
//             error_code ec;
//             naming::id_type console_prefix;
//             if (agas_client_.get_console_prefix(console_prefix, ec))
//             {
//                 util::section ini;
//                 components::stubs::runtime_support::get_config(console_prefix, ini);
//                 ini_.add_section("application", ini);
//             }
//         }

        // now, execute the user supplied thread function
        if (!func.empty()) 
            result = func();
        return threads::thread_state(threads::terminated);
    }
#else
    void pre_main();

    template <typename SchedulingPolicy, typename NotificationPolicy> 
    threads::thread_state 
    runtime_impl<SchedulingPolicy, NotificationPolicy>::run_helper(
        boost::function<runtime::hpx_main_function_type> func, int& result)
    {
        ::hpx::pre_main();

        // now, execute the user supplied thread function
        if (!func.empty()) 
            result = func();
        return threads::thread_state(threads::terminated);
    }
#endif

    template <typename SchedulingPolicy, typename NotificationPolicy> 
    int runtime_impl<SchedulingPolicy, NotificationPolicy>::start(
        boost::function<hpx_main_function_type> func, 
        std::size_t num_threads, std::size_t num_localities, bool blocking)
    {
#if defined(_WIN64) && defined(_DEBUG) && !defined(BOOST_COROUTINE_USE_FIBERS)
        // needs to be called to avoid problems at system startup
        // see: http://connect.microsoft.com/VisualStudio/feedback/ViewFeedback.aspx?FeedbackID=100319
        _isatty(0);
#endif
        // {{{ early startup code
#if HPX_AGAS_VERSION <= 0x10
        // init TSS for the main thread, this enables logging, time logging, etc.
        init_tss();
#else
        // in AGAS v2, the runtime pointer (accessible through get_runtime
        // and get_runtime_ptr) is already initialized at this point.
        applier_.init_tss();
#endif

        LRT_(info) << "runtime_impl: beginning startup sequence";

        LRT_(info) << "runtime_impl: starting services";
        // start services (service threads)
        runtime_support_.run();
        LRT_(info) << "runtime_impl: started runtime_support component";

        // AGAS v2 starts the parcel port itself
#if HPX_AGAS_VERSION <= 0x10
        parcel_port_.run(false);            // starts parcel_pool_ as well
        LRT_(info) << "runtime_impl: started parcelport";
#endif

#if HPX_AGAS_VERSION > 0x10
        io_pool_.run(false); // start io pool
#endif

        thread_manager_.run(num_threads);   // start the thread manager, timer_pool_ as well
        LRT_(info) << "runtime_impl: started threadmanager";
        // }}}

        // AGAS v2 handles this in the client
#if HPX_AGAS_VERSION <= 0x10
        // {{{ exiting bootstrap mode 
        LRT_(info) << "runtime_impl: registering runtime_support and memory components";
        // register the runtime_support and memory instances with the AGAS 
        agas_client_.bind(applier_.get_runtime_support_raw_gid(), 
            naming::address(parcel_port_.here(), 
                components::get_component_type<components::server::runtime_support>(), 
                &runtime_support_));

        agas_client_.bind(applier_.get_memory_raw_gid(), 
            naming::address(parcel_port_.here(), 
                components::get_component_type<components::server::memory>(), 
                &memory_));
#endif
        // }}}

        // {{{ late startup - distributed
        // if there are more than one localities involved, wait for all
        // to get registered
#if HPX_AGAS_VERSION <= 0x10
        if (num_localities > 1) {
            bool foundall = false;
            for (int i = 0; i < HPX_MAX_NETWORK_RETRIES; ++i) {
                std::vector<naming::gid_type> prefixes;
                error_code ec;
                // NOTE: in AGAS v2, AGAS enforces a distributed, global barrier
                // before we get here, so this should always succeed 
                if (agas_client_.get_prefixes(prefixes, ec) &&
                    num_localities == prefixes.size()) 
                {
                    foundall = true;
                    break;
                }

                boost::this_thread::sleep(boost::get_system_time() + 
                    boost::posix_time::milliseconds(HPX_NETWORK_RETRIES_SLEEP));
            } 
            if (!foundall) {
                HPX_THROW_EXCEPTION(startup_timed_out, "runtime::run", 
                    "timed out while waiting for other localities");
            }
        }
#else
        // invoke the AGAS v2 notifications, waking up the other localities
        agas::get_big_boot_barrier().trigger();  
#endif

#if HPX_AGAS_VERSION <= 0x10
        LRT_(info) << "runtime_impl: setting initial locality prefixes";
        // Set localities prefixes once per runtime instance. This should
        // work fine until we support adding and removing localities.
        {
            std::size_t here_lid = std::size_t(-1);
            naming::gid_type tmp_here(applier_.get_runtime_support_raw_gid());

            std::vector<naming::gid_type> tmp_localities;
            agas_client_.get_prefixes(tmp_localities);

            for (std::size_t i = 0; i < tmp_localities.size(); ++i)
            {
                if (tmp_here.get_msb() == tmp_localities[i].get_msb())
                {
                    here_lid = i;
                    break;
                }
            }

            if (here_lid == std::size_t(-1))
            {
                hpx::util::osstream strm;
                strm << "failed to find prefix of this locality: " 
                     << tmp_here;
                HPX_THROW_EXCEPTION(startup_timed_out, "runtime::run", 
                    hpx::util::osstream_get_string(strm));
            }

            // wrap all gid_types into id_types
            std::vector<naming::id_type> prefixes;
            BOOST_FOREACH(naming::gid_type& gid, tmp_localities)
                prefixes.push_back(naming::id_type(gid, naming::id_type::unmanaged));

            this->process_.set_num_os_threads(num_threads);
            this->process_.set_localities(here_lid, prefixes);
        }
#else
        this->process_.set_num_os_threads(num_threads);
#endif
        // }}}

        // {{{ launch main 
        // register the given main function with the thread manager
        threads::thread_init_data data(
            boost::bind(&runtime_impl::run_helper, this, func, boost::ref(result_)), 
            "hpx_main");
        thread_manager_.register_thread(data);
        this->runtime::start();

        LRT_(info) << "runtime_impl: started using "  << num_threads << " OS threads";
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
        boost::function<hpx_main_function_type> empty_main;
        return start(empty_main, num_threads, num_localities, blocking);
    }

    ///////////////////////////////////////////////////////////////////////////
// #if !defined(BOOST_WINDOWS)
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
// #endif

    template <typename SchedulingPolicy, typename NotificationPolicy> 
    int runtime_impl<SchedulingPolicy, NotificationPolicy>::wait()
    {
        LRT_(info) << "runtime_impl: about to enter wait state";

// #if defined(BOOST_WINDOWS)
//         // Set console control handler to allow server to be stopped.
//         console_ctrl_function = boost::bind(&runtime_impl::stop, this, true);
//         SetConsoleCtrlHandler(console_ctrl_handler, TRUE);
// 
//         // wait for the shutdown action to be executed
//         runtime_support_.wait();
// #else
//         // Block all signals for background thread.
//         sigset_t new_mask;
//         sigfillset(&new_mask);
//         sigset_t old_mask;
//         pthread_sigmask(SIG_BLOCK, &new_mask, &old_mask);

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

//         // Restore previous signals.
//         pthread_sigmask(SIG_SETMASK, &old_mask, 0);
// 
//         // Wait for signal indicating time to shut down.
//         sigset_t wait_mask;
//         sigemptyset(&wait_mask);
//         sigaddset(&wait_mask, SIGINT);
//         sigaddset(&wait_mask, SIGQUIT);
//         sigaddset(&wait_mask, SIGTERM);
//         pthread_sigmask(SIG_BLOCK, &wait_mask, 0);
//         int sig = 0;
//         sigwait(&wait_mask, &sig);

        // block main thread
        t.join();
// #endif
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

        #if HPX_AGAS_VERSION <= 0x10
            parcel_port_.get_io_service_pool().get_io_service().post
        #else
            boost::thread t
        #endif
            (boost::bind(&runtime_impl::stopped, this, blocking, 
                boost::ref(cond), boost::ref(mtx)));
        cond.wait(l);

        #if HPX_AGAS_VERSION > 0x10
            t.join();
        #endif

//        // stop the rest of the system
//        parcel_port_.stop(blocking);        // stops parcel_pool_ as well

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

        // unregister the runtime_support and memory instances from the AGAS 
        // ignore errors, as AGAS might be down already
        #if HPX_AGAS_VERSION <= 0x10
            error_code ec;
            agas_client_.unbind(applier_.get_runtime_support_raw_gid(), ec);
            agas_client_.unbind(applier_.get_memory_raw_gid(), ec);
        #endif

        // this disables all logging from the main thread
        deinit_tss();

        LRT_(info) << "runtime_impl: stopped all services";

        boost::mutex::scoped_lock l(mtx);
        cond.notify_all();                  // we're done now
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy, typename NotificationPolicy> 
    void runtime_impl<SchedulingPolicy, NotificationPolicy>::report_error(
        std::size_t num_thread, 
        boost::exception_ptr const& e)
    {
        // The console error sink is only applied at the console, so default
        // error sink never gets called on the locality, meaning that the user
        // never sees errors that kill the system before the error parcel gets
        // sent out. So, before we try to send the error parcel (which might
        // cause a double fault), print local diagnostics.
        components::server::console_error_sink
            (naming::get_prefix_from_gid(parcel_handler_.get_prefix()), e);

        // first report this error to the console
        naming::gid_type console_prefix;
        if (agas_client_.get_console_prefix(console_prefix))
        {
            if (parcel_handler_.get_prefix() != console_prefix)
                components::console_error_sink(
                    naming::id_type(console_prefix, naming::id_type::unmanaged), 
                    parcel_handler_.get_prefix(), e);
        }

        // stop all services
        stop(false);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy, typename NotificationPolicy> 
    int runtime_impl<SchedulingPolicy, NotificationPolicy>::run(
        boost::function<hpx_main_function_type> func,
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
        boost::uint32_t src, std::string const& msg)
    {
        boost::io::ios_all_saver ifs(std::cerr); 
        std::cerr << "locality (" << std::hex << std::setw(4) 
                  << std::setfill('0') << src << "):" << std::endl
                  << msg << std::endl;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy, typename NotificationPolicy> 
    void runtime_impl<SchedulingPolicy, NotificationPolicy>::init_tss()
    {
        // initialize our TSS
        this->runtime::init_tss();

        // initialize applier TSS
        applier_.init_tss();
    }

    template <typename SchedulingPolicy, typename NotificationPolicy> 
    void runtime_impl<SchedulingPolicy, NotificationPolicy>::deinit_tss()
    {
        // reset applier TSS
        applier_.deinit_tss();

        // reset our TSS
        this->runtime::deinit_tss();
    }

    template <typename SchedulingPolicy, typename NotificationPolicy> 
    naming::gid_type
    runtime_impl<SchedulingPolicy, NotificationPolicy>::get_next_id()
    {
        return id_pool.get_id(parcel_port_.here(), agas_client_);
    }

    ///////////////////////////////////////////////////////////////////////////
    boost::thread_specific_ptr<runtime *> runtime::runtime_;

    void runtime::init_tss()
    {
        // initialize our TSS
        BOOST_ASSERT(NULL == runtime::runtime_.get());    // shouldn't be initialized yet
        runtime::runtime_.reset(new runtime* (this));
    }

    void runtime::deinit_tss()
    {
        // reset our TSS
        runtime::runtime_.reset();
    }

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
    { return get_runtime().here(); }

    void report_error(
        std::size_t num_thread
      , boost::exception_ptr const& e
    ) {
        get_runtime().report_error(num_thread, e);
    }

    void report_error(
        boost::exception_ptr const& e
    ) {
        get_runtime().report_error(e);
    }

    bool register_on_exit(boost::function<void()> f)
    {
        runtime* rt = get_runtime_ptr();
        if (NULL == rt)
            return false;
        rt->on_exit(f);
        return true;
    }

    std::size_t get_runtime_instance_number()
    {
//         runtime* rt = get_runtime_ptr();
//         return (NULL == rt) ? 0 : rt->get_instance_number();
        return get_runtime().get_instance_number();
    }

    std::string get_config_entry(std::string const& key, std::string const& dflt)
    {
        return get_runtime().get_config().get_entry(key, dflt);
    }

    std::string get_config_entry(std::string const& key, std::size_t dflt)
    {
        return get_runtime().get_config().get_entry(key, dflt);
    }

    ///////////////////////////////////////////////////////////////////////////
    // Helpers
    naming::id_type find_here()
    {
        return naming::id_type(applier::get_applier().get_prefix()
                             , naming::id_type::unmanaged);
    }

    naming::gid_type get_next_id()
    {
        return get_runtime().get_next_id();
    }
}

///////////////////////////////////////////////////////////////////////////////
/// explicit template instantiation for the thread manager of our choice
template HPX_EXPORT class hpx::runtime_impl<
    hpx::threads::policies::global_queue_scheduler, 
    hpx::threads::policies::callback_notifier>;

template HPX_EXPORT class hpx::runtime_impl<
    hpx::threads::policies::local_queue_scheduler, 
    hpx::threads::policies::callback_notifier>;

template HPX_EXPORT class hpx::runtime_impl<
    hpx::threads::policies::local_priority_queue_scheduler, 
    hpx::threads::policies::callback_notifier>;

template HPX_EXPORT class hpx::runtime_impl<
    hpx::threads::policies::abp_queue_scheduler, 
    hpx::threads::policies::callback_notifier>;

