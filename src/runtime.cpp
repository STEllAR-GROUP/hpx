//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <vector>

#include <hpx/hpx_fwd.hpp>

#include <boost/config.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>

#include <hpx/include/runtime.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/runtime/components/console_error_sink.hpp>
#include <hpx/runtime/components/runtime_support.hpp>

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

#include <pthread.h>
#include <signal.h>

#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx 
{
    ///////////////////////////////////////////////////////////////////////////
    boost::detail::atomic_count runtime::instance_number_counter_(-1);

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy, typename NotificationPolicy> 
    runtime_impl<SchedulingPolicy, NotificationPolicy>::runtime_impl(
            std::string const& address, boost::uint16_t port,
            std::string const& agas_address, boost::uint16_t agas_port, 
            mode locality_mode, init_scheduler_type const& init) 
      : runtime(agas_client_),
        result_(0), mode_(locality_mode), 
        agas_pool_(boost::bind(&runtime_impl::init_tss, This()),
            boost::bind(&runtime_impl::deinit_tss, This())), 
        parcel_pool_(boost::bind(&runtime_impl::init_tss, This()),
            boost::bind(&runtime_impl::deinit_tss, This())), 
        timer_pool_(boost::bind(&runtime_impl::init_tss, This()),
            boost::bind(&runtime_impl::deinit_tss, This())),
        agas_client_(agas_pool_, ini_.get_agas_locality(agas_address, agas_port), 
            ini_.get_agas_smp_mode(), mode_ == console, ini_.get_agas_cache_size()),
        parcel_port_(parcel_pool_, naming::locality(address, port)),
        parcel_handler_(agas_client_, parcel_port_, &thread_manager_),
        init_logging_(ini_, mode_ == console, agas_client_),
        scheduler_(init),
        notifier_(boost::bind(&runtime_impl::init_tss, This()),
            boost::bind(&runtime_impl::deinit_tss, This()),
            boost::bind(&runtime_impl::report_error, This(), _1, _2)),
        thread_manager_(timer_pool_, scheduler_, notifier_),
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
            mode locality_mode, init_scheduler_type const& init) 
      : runtime(agas_client_),
        result_(0), mode_(locality_mode), 
        agas_pool_(boost::bind(&runtime_impl::init_tss, This()),
            boost::bind(&runtime_impl::deinit_tss, This())), 
        parcel_pool_(boost::bind(&runtime_impl::init_tss, This()),
            boost::bind(&runtime_impl::deinit_tss, This())), 
        timer_pool_(boost::bind(&runtime_impl::init_tss, This()),
            boost::bind(&runtime_impl::deinit_tss, This())),
        agas_client_(agas_pool_, agas_address, ini_.get_agas_smp_mode(),
            mode_ == console, ini_.get_agas_cache_size()),
        parcel_port_(parcel_pool_, address),
        parcel_handler_(agas_client_, parcel_port_, &thread_manager_),
        init_logging_(ini_, mode_ == console, agas_client_),
        scheduler_(init),
        notifier_(boost::bind(&runtime_impl::init_tss, This()),
            boost::bind(&runtime_impl::deinit_tss, This()),
            boost::bind(&runtime_impl::report_error, This(), _1, _2)),
        thread_manager_(timer_pool_, scheduler_, notifier_),
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
            naming::locality address, mode locality_mode, 
            init_scheduler_type const& init) 
      : runtime(agas_client_),
        result_(0), mode_(locality_mode), 
        agas_pool_(boost::bind(&runtime_impl::init_tss, This()),
            boost::bind(&runtime_impl::deinit_tss, This())), 
        parcel_pool_(boost::bind(&runtime_impl::init_tss, This()),
            boost::bind(&runtime_impl::deinit_tss, This())), 
        timer_pool_(boost::bind(&runtime_impl::init_tss, This()),
            boost::bind(&runtime_impl::deinit_tss, This())),
        agas_client_(agas_pool_, ini_.get_agas_locality(), 
            ini_.get_agas_smp_mode(), mode_ == console, ini_.get_agas_cache_size()),
        parcel_port_(parcel_pool_, address),
        parcel_handler_(agas_client_, parcel_port_, &thread_manager_),
        init_logging_(ini_, mode_ == console, agas_client_),
        scheduler_(init),
        notifier_(boost::bind(&runtime_impl::init_tss, This()),
            boost::bind(&runtime_impl::deinit_tss, This()),
            boost::bind(&runtime_impl::report_error, This(), _1, _2)),
        thread_manager_(timer_pool_, scheduler_, notifier_),
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
        agas_pool_.stop();

        // unload libraries
        runtime_support_.tidy();

        LRT_(debug) << "~runtime_impl(finished)";
    }

    ///////////////////////////////////////////////////////////////////////////
#if defined(_WIN64) && defined(_DEBUG) && !defined(BOOST_COROUTINE_USE_FIBERS)
#include <io.h>
#endif

    ///////////////////////////////////////////////////////////////////////////
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
        // init TSS for the main thread, this enables logging, time logging, etc.
        init_tss();

        // start services (service threads)
        runtime_support_.run();
        parcel_port_.run(false);            // starts parcel_pool_ as well
        thread_manager_.run(num_threads);   // start the thread manager, timer_pool_ as well

        // register the runtime_support and memory instances with the AGAS 
        agas_client_.bind(applier_.get_runtime_support_raw_gid(), 
            naming::address(parcel_port_.here(), 
                components::get_component_type<components::server::runtime_support>(), 
                &runtime_support_));

        agas_client_.bind(applier_.get_memory_raw_gid(), 
            naming::address(parcel_port_.here(), 
                components::get_component_type<components::server::memory>(), 
                &memory_));

        // if there are more than one localities involved, wait for all
        // to get registered
        if (num_localities > 1) {
            bool foundall = false;
            for (int i = 0; i < HPX_MAX_NETWORK_RETRIES; ++i) {
                std::vector<naming::gid_type> prefixes;
                error_code ec;
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

        // register the given main function with the thread manager
        threads::thread_init_data data(
            boost::bind(&runtime_impl::run_helper, this, func, boost::ref(result_)), 
            "hpx_main");
        thread_manager_.register_thread(data);

        LRT_(info) << "runtime_impl: started using "  << num_threads << " OS threads";

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
#if !defined(BOOST_WINDOWS)
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
#endif

    template <typename SchedulingPolicy, typename NotificationPolicy> 
    int runtime_impl<SchedulingPolicy, NotificationPolicy>::wait()
    {
        LRT_(info) << "runtime_impl: about to enter wait state";

#if defined(BOOST_WINDOWS)
        // Set console control handler to allow server to be stopped.
        console_ctrl_function = boost::bind(&runtime_impl::stop, this, true);
        SetConsoleCtrlHandler(console_ctrl_handler, TRUE);

        // wait for the shutdown action to be executed
        runtime_support_.wait();
#else
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
#endif
        LRT_(info) << "runtime_impl: exiting wait state";
        return result_;
    }

    ///////////////////////////////////////////////////////////////////////////
    // First half of termination process: stop thread manager,
    // schedule a task managed by timer_pool to initiate second part
    template <typename SchedulingPolicy, typename NotificationPolicy> 
    void runtime_impl<SchedulingPolicy, NotificationPolicy>::stop(bool blocking)
    {
        LRT_(info) << "runtime_impl: about to stop services";

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
        parcel_port_.get_io_service_pool().get_io_service().post(
            boost::bind(&runtime_impl::stopped, this, blocking, boost::ref(cond)));
        cond.wait(l);

        // stop the rest of the system
        parcel_port_.stop(blocking);        // stops parcel_pool_ as well

        deinit_tss();
    }

    // Second step in termination: shut down all services.
    // This gets executed as a task in the timer_pool io_service and not as 
    // a PX thread!
    template <typename SchedulingPolicy, typename NotificationPolicy> 
    void runtime_impl<SchedulingPolicy, NotificationPolicy>::stopped(
        bool blocking, boost::condition& cond)
    {
        // unregister the runtime_support and memory instances from the AGAS 
        // ignore errors, as AGAS might be down already
        error_code ec;
        agas_client_.unbind(applier_.get_runtime_support_raw_gid(), ec);
        agas_client_.unbind(applier_.get_memory_raw_gid(), ec);

        // wait for thread manager to exit
        runtime_support_.stopped();         // re-activate main thread 
        thread_manager_.stop(blocking);     // wait for thread manager

        // this disables all logging from the main thread
        deinit_tss();

        LRT_(info) << "runtime_impl: stopped all services";

        // we're done now
        cond.notify_all();
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy, typename NotificationPolicy> 
    void runtime_impl<SchedulingPolicy, NotificationPolicy>::report_error(
        std::size_t num_thread, 
        boost::exception_ptr const& e)
    {
        // first report this error to the console
        naming::gid_type console_prefix;
        if (agas_client_.get_console_prefix(console_prefix))
        {
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
        return result;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy, typename NotificationPolicy> 
    void runtime_impl<SchedulingPolicy, NotificationPolicy>::default_errorsink(
        boost::uint32_t src, std::string const& msg)
    {
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
}

///////////////////////////////////////////////////////////////////////////////
/// explicit template instantiation for the thread manager of our choice
template HPX_EXPORT class hpx::runtime_impl<
    hpx::threads::policies::global_queue_scheduler, 
    hpx::threads::policies::callback_notifier>;

template HPX_EXPORT class hpx::runtime_impl<
    hpx::threads::policies::local_queue_scheduler, 
    hpx::threads::policies::callback_notifier>;
