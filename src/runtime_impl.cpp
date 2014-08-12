//  Copyright (c) 2007-2013 Hartmut Kaiser
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
#include <hpx/runtime_impl.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/stringstream.hpp>
#include <hpx/util/set_thread_name.hpp>
#include <hpx/util/thread_mapper.hpp>
#include <hpx/util/apex.hpp>
#include <hpx/runtime/components/console_error_sink.hpp>
#include <hpx/runtime/components/server/console_error_sink.hpp>
#include <hpx/runtime/components/runtime_support.hpp>
#include <hpx/runtime/parcelset/policies/global_parcelhandler_queue.hpp>
#include <hpx/runtime/threads/threadmanager_impl.hpp>
#include <hpx/include/performance_counters.hpp>
#include <hpx/runtime/agas/big_boot_barrier.hpp>

#if defined(_WIN64) && defined(_DEBUG) && !defined(HPX_HAVE_FIBER_BASED_COROUTINES)
#include <io.h>
#endif

#if defined(HPX_HAVE_PARCELPORT_MPI)
#include <hpx/util/mpi_environment.hpp>
#endif

namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    // There is no need to protect these global from thread concurrent access
    // as they are access during early startup only.
    std::list<HPX_STD_FUNCTION<void()> > global_pre_startup_functions;
    std::list<HPX_STD_FUNCTION<void()> > global_startup_functions;

    std::list<HPX_STD_FUNCTION<void()> > global_pre_shutdown_functions;
    std::list<HPX_STD_FUNCTION<void()> > global_shutdown_functions;

    ///////////////////////////////////////////////////////////////////////////
    void register_pre_startup_function(startup_function_type const& f)
    {
        runtime* rt = get_runtime_ptr();
        if (NULL != rt) {
            if (rt->get_state() > runtime::state_pre_startup) {
                HPX_THROW_EXCEPTION(invalid_status,
                    "register_pre_startup_function",
                    "Too late to register a new pre-startup function.");
                return;
            }
            rt->add_pre_startup_function(f);
        }
        else {
            global_pre_startup_functions.push_back(f);
        }
    }

    void register_startup_function(startup_function_type const& f)
    {
        runtime* rt = get_runtime_ptr();
        if (NULL != rt) {
            if (rt->get_state() > runtime::state_startup) {
                HPX_THROW_EXCEPTION(invalid_status,
                    "register_startup_function",
                    "Too late to register a new startup function.");
                return;
            }
            rt->add_startup_function(f);
        }
        else {
            global_startup_functions.push_back(f);
        }
    }

    void register_pre_shutdown_function(shutdown_function_type const& f)
    {
        runtime* rt = get_runtime_ptr();
        if (NULL != rt) {
            if (rt->get_state() > runtime::state_pre_shutdown) {
                HPX_THROW_EXCEPTION(invalid_status,
                    "register_pre_shutdown_function",
                    "Too late to register a new pre-shutdown function.");
                return;
            }
            rt->add_pre_shutdown_function(f);
        }
        else {
            global_pre_shutdown_functions.push_back(f);
        }
    }

    void register_shutdown_function(shutdown_function_type const& f)
    {
        runtime* rt = get_runtime_ptr();
        if (NULL != rt) {
            if (rt->get_state() > runtime::state_shutdown) {
                HPX_THROW_EXCEPTION(invalid_status,
                    "register_shutdown_function",
                    "Too late to register a new shutdown function.");
                return;
            }
            rt->add_shutdown_function(f);
        }
        else {
            global_shutdown_functions.push_back(f);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy, typename NotificationPolicy>
    runtime_impl<SchedulingPolicy, NotificationPolicy>::runtime_impl(
            util::runtime_configuration const& rtcfg,
            runtime_mode locality_mode, std::size_t num_threads,
            init_scheduler_type const& init,
            threads::policies::init_affinity_data const& init_affinity)
      : runtime(rtcfg, init_affinity),
        mode_(locality_mode), result_(0), num_threads_(num_threads),
        main_pool_(1,
            boost::bind(&runtime_impl::init_tss, This(), "main-thread", ::_1, ::_2, false),
            boost::bind(&runtime_impl::deinit_tss, This()), "main_pool"),
        io_pool_(rtcfg.get_thread_pool_size("io_pool"),
            boost::bind(&runtime_impl::init_tss, This(), "io-thread", ::_1, ::_2, true),
            boost::bind(&runtime_impl::deinit_tss, This()), "io_pool"),
        timer_pool_(rtcfg.get_thread_pool_size("timer_pool"),
            boost::bind(&runtime_impl::init_tss, This(), "timer-thread", ::_1, ::_2, true),
            boost::bind(&runtime_impl::deinit_tss, This()), "timer_pool"),
        parcel_port_(parcelset::parcelport::create_bootstrap(ini_,
            boost::bind(&runtime_impl::init_tss, This(), "parcel-thread", ::_1, ::_2, true),
            boost::bind(&runtime_impl::deinit_tss, This()))),
        scheduler_(init),
        notifier_(
            boost::bind(&runtime_impl::init_tss, This(), "worker-thread", ::_1, ::_2, false),
            boost::bind(&runtime_impl::deinit_tss, This()),
            boost::bind(&runtime_impl::report_error, This(), _1, _2)),
        thread_manager_(new hpx::threads::threadmanager_impl<
            SchedulingPolicy, NotificationPolicy>(
                timer_pool_, scheduler_, notifier_, num_threads)),
        agas_client_(*parcel_port_, ini_, mode_),
        parcel_handler_(agas_client_, thread_manager_.get(),
            new parcelset::policies::global_parcelhandler_queue),
        init_logging_(ini_, mode_ == runtime_mode_console, agas_client_),
        applier_(parcel_handler_, *thread_manager_,
            boost::uint64_t(runtime_support_.get()), boost::uint64_t(memory_.get())),
        action_manager_(applier_)
    {
        components::server::get_error_dispatcher().register_error_sink(
            &runtime_impl::default_errorsink, default_error_sink_);

        // in AGAS v2, the runtime pointer (accessible through get_runtime
        // and get_runtime_ptr) is already initialized at this point.
        applier_.init_tss();

#if defined(HPX_HAVE_SECURITY)
        // once all has been initialized, finalize security data for bootstrap
        this->init_security();
#endif
        // now, launch AGAS and register all nodes, launch all other components
        agas_client_.initialize(*parcel_port_);
        parcel_handler_.initialize(parcel_port_);

#if defined(HPX_HAVE_SECURITY)
        // enable parcel capability checking
        applier_.enable_verify_capabilities();
#endif

        // copy over all startup functions registered so far
        BOOST_FOREACH(HPX_STD_FUNCTION<void()> const& f, global_pre_startup_functions)
        {
            add_pre_startup_function(f);
        }
        global_pre_startup_functions.clear();

        BOOST_FOREACH(HPX_STD_FUNCTION<void()> const& f, global_startup_functions)
        {
            add_startup_function(f);
        }
        global_startup_functions.clear();

        BOOST_FOREACH(HPX_STD_FUNCTION<void()> const& f, global_pre_shutdown_functions)
        {
            add_pre_shutdown_function(f);
        }
        global_pre_shutdown_functions.clear();

        BOOST_FOREACH(HPX_STD_FUNCTION<void()> const& f, global_shutdown_functions)
        {
            add_shutdown_function(f);
        }
        global_shutdown_functions.clear();

        // set state to initialized
        set_state(state_initialized);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy, typename NotificationPolicy>
    runtime_impl<SchedulingPolicy, NotificationPolicy>::~runtime_impl()
    {
        LRT_(debug) << "~runtime_impl(entering)";

        // stop all services
        parcel_handler_.stop();     // stops parcel pools as well
        thread_manager_->stop();    // stops timer_pool_ as well
        io_pool_.stop();

        // unload libraries
        //runtime_support_->tidy();

#if defined(HPX_HAVE_PARCELPORT_MPI)
        util::mpi_environment::finalize();
#endif

        LRT_(debug) << "~runtime_impl(finished)";
    }

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
            LBT_(info) << "runtime_impl::run_helper: bootstrap "
                          "aborted, bailing out";
            return threads::thread_state(threads::terminated);
        }

        LBT_(info) << "(4th stage) runtime_impl::run_helper: bootstrap complete";
        set_state(state_running);

        parcel_handler_.enable_alternative_parcelports();

        // Now, execute the user supplied thread function (hpx_main)
        if (!!func) {
            // Change our thread description, as we're about to call hpx_main
            threads::set_thread_description(threads::get_self_id(), "hpx_main");

            // Call hpx_main
            result = func();
        }

        return threads::thread_state(threads::terminated);
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    int runtime_impl<SchedulingPolicy, NotificationPolicy>::start(
        HPX_STD_FUNCTION<hpx_main_function_type> const& func, bool blocking)
    {
#if defined(_WIN64) && defined(_DEBUG) && !defined(HPX_HAVE_FIBER_BASED_COROUTINES)
        // needs to be called to avoid problems at system startup
        // see: http://connect.microsoft.com/VisualStudio/feedback/ViewFeedback.aspx?FeedbackID=100319
        _isatty(0);
#endif
        // {{{ early startup code - local

        // initialize instrumentation system
        util::apex_init();

        LRT_(info) << "cmd_line: " << get_config().get_cmd_line();

        LBT_(info) << "(1st stage) runtime_impl::start: booting locality " //-V128
                   << here() << " on " << num_threads_ << " OS-thread"
                   << ((num_threads_ == 1) ? "" : "s");

        // start runtime_support services
        runtime_support_->run();
        LBT_(info) << "(1st stage) runtime_impl::start: started "
                      "runtime_support component";

        // start the io pool
        io_pool_.run(false);
        LBT_(info) << "(1st stage) runtime_impl::start: started the application "
                      "I/O service pool";

        // start the thread manager
        thread_manager_->run(num_threads_);
        LBT_(info) << "(1st stage) runtime_impl::start: started threadmanager";
        // }}}

        // invoke the AGAS v2 notifications
        agas::get_big_boot_barrier().trigger();

        // {{{ launch main
        // register the given main function with the thread manager
        LBT_(info) << "(1st stage) runtime_impl::start: launching run_helper "
                      "HPX thread";

        threads::thread_init_data data(
            boost::bind(&runtime_impl::run_helper, this, func,
                boost::ref(result_)),
            "run_helper", 0, threads::thread_priority_normal, std::size_t(-1),
            threads::get_stack_size(threads::thread_stacksize_large));
        thread_manager_->register_thread(data);
        this->runtime::starting();
        // }}}

        // block if required
        if (blocking)
            return wait();     // wait for the shutdown_action to be executed

        // Register this thread with the runtime system to allow calling certain
        // HPX functionality from the main thread.
        init_tss("main-thread", 0, "", false);

        return 0;   // return zero as we don't know the outcome of hpx_main yet
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    int runtime_impl<SchedulingPolicy, NotificationPolicy>::start(bool blocking)
    {
        HPX_STD_FUNCTION<hpx_main_function_type> empty_main;
        return start(empty_main, blocking);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy, typename NotificationPolicy>
    void runtime_impl<SchedulingPolicy, NotificationPolicy>::wait_helper(
        boost::mutex& mtx, boost::condition& cond, bool& running)
    {
        // signal successful initialization
        {
            boost::mutex::scoped_lock lk(mtx);
            running = true;
            cond.notify_all();
        }

        // register this thread with any possibly active Intel tool
        HPX_ITT_THREAD_SET_NAME("main-thread#wait_helper");

        // set thread name as shown in Visual Studio
        util::set_thread_name("main-thread#wait_helper");

        // wait for termination
        runtime_support_->wait();

        // stop main thread pool
        main_pool_.stop();
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
                &runtime_impl<SchedulingPolicy, NotificationPolicy>::wait_helper,
                this, boost::ref(mtx), boost::ref(cond), boost::ref(running)
            ));

        // wait for the thread to run
        {
            boost::mutex::scoped_lock lk(mtx);
            while (!running)
                cond.wait(lk);
        }

        // use main thread to drive main thread pool
        main_pool_.thread_run(0);

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

        // flush all parcel buffers, stop buffering parcels at this point
        parcel_handler_.do_background_work(true);

        // execute all on_exit functions whenever the first thread calls this
        this->runtime::stopping();

        // stop runtime_impl services (threads)
        thread_manager_->stop(false);    // just initiate shutdown

        // schedule task in timer_pool to execute stopped() below
        // this is necessary as this function (stop()) might have been called
        // from a HPX thread, so it would deadlock by waiting for the thread
        // manager
        boost::mutex mtx;
        boost::condition cond;
        boost::mutex::scoped_lock l(mtx);

        boost::thread t(boost::bind(&runtime_impl::stopped, this, blocking,
            boost::ref(cond), boost::ref(mtx)));
        cond.wait(l);

        t.join();

        // stop the rest of the system
        parcel_handler_.stop(blocking);     // stops parcel pools as well
        io_pool_.stop();                    // stops io_pool_ as well

        deinit_tss();
    }

    // Second step in termination: shut down all services.
    // This gets executed as a task in the timer_pool io_service and not as
    // a HPX thread!
    template <typename SchedulingPolicy, typename NotificationPolicy>
    void runtime_impl<SchedulingPolicy, NotificationPolicy>::stopped(
        bool blocking, boost::condition& cond, boost::mutex& mtx)
    {
        // wait for thread manager to exit
        runtime_support_->stopped();         // re-activate shutdown HPX-thread
        thread_manager_->stop(blocking);     // wait for thread manager

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
        // Early and late exceptions, errors outside of HPX-threads
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
        HPX_STD_FUNCTION<hpx_main_function_type> const& func)
    {
        // start the main thread function
        start(func);

        // now wait for everything to finish
        wait();
        stop();

        parcel_handler_.stop();      // stops parcelport for sure
        return result_;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy, typename NotificationPolicy>
    int runtime_impl<SchedulingPolicy, NotificationPolicy>::run()
    {
        // start the main thread function
        start();

        // now wait for everything to finish
        int result = wait();
        stop();

        parcel_handler_.stop();      // stops parcelport for sure
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
        char const* context, std::size_t num, char const* postfix,
        bool service_thread)
    {
        // initialize our TSS
        this->runtime::init_tss();

        // initialize applier TSS
        applier_.init_tss();

        // set the thread's name, if it's not already set
        if (NULL == runtime::thread_name_.get())
        {
            std::string* fullname = new std::string(context);
            if (postfix && *postfix)
                *fullname += postfix;
            *fullname += "#" + boost::lexical_cast<std::string>(num);
            runtime::thread_name_.reset(fullname);

            char const* name = runtime::thread_name_.get()->c_str();

            // initialize thread mapping for external libraries (i.e. PAPI)
            thread_support_->register_thread(name);

            // initialize coroutines context switcher
            hpx::util::coroutines::thread_startup(name);

            // register this thread with any possibly active Intel tool
            HPX_ITT_THREAD_SET_NAME(name);

            // set thread name as shown in Visual Studio
            util::set_thread_name(name);
        }

        // if this is a service thread, set its service affinity
        if (service_thread)
        {
            // FIXME: We don't set the affinity of the service threads on BG/Q, as this is
            // causing a hang (needs to be investigated
#if !defined(__bgq__)
            threads::mask_cref_type used_processing_units =
                thread_manager_->get_used_processing_units();

            this->topology_.set_thread_affinity_mask(
                this->topology_.get_service_affinity_mask(
                    used_processing_units));
#endif
        }
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    void runtime_impl<SchedulingPolicy, NotificationPolicy>::deinit_tss()
    {
        // initialize coroutines context switcher
        hpx::util::coroutines::thread_shutdown();

        // reset applier TSS
        applier_.deinit_tss();

        // reset our TSS
        this->runtime::deinit_tss();

        // reset PAPI support
        thread_support_->unregister_thread();

        // reset thread local storage
        runtime::thread_name_.reset();
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    naming::gid_type
    runtime_impl<SchedulingPolicy, NotificationPolicy>::get_next_id(std::size_t count)
    {
        return id_pool_.get_id(count);
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    void runtime_impl<SchedulingPolicy, NotificationPolicy>::
        add_pre_startup_function(HPX_STD_FUNCTION<void()> const& f)
    {
        runtime_support_->add_pre_startup_function(f);
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    void runtime_impl<SchedulingPolicy, NotificationPolicy>::
        add_startup_function(HPX_STD_FUNCTION<void()> const& f)
    {
        runtime_support_->add_startup_function(f);
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    void runtime_impl<SchedulingPolicy, NotificationPolicy>::
        add_pre_shutdown_function(HPX_STD_FUNCTION<void()> const& f)
    {
        runtime_support_->add_pre_shutdown_function(f);
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    void runtime_impl<SchedulingPolicy, NotificationPolicy>::
        add_shutdown_function(HPX_STD_FUNCTION<void()> const& f)
    {
        runtime_support_->add_shutdown_function(f);
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    bool runtime_impl<SchedulingPolicy, NotificationPolicy>::
        keep_factory_alive(components::component_type type)
    {
        return runtime_support_->keep_factory_alive(type);
    }

    template <typename SchedulingPolicy, typename NotificationPolicy>
    hpx::util::io_service_pool*
    runtime_impl<SchedulingPolicy, NotificationPolicy>::
        get_thread_pool(char const* name)
    {
        HPX_ASSERT(name != 0);

        if (0 == std::strncmp(name, "io", 2))
            return &io_pool_;
        if (0 == std::strncmp(name, "parcel", 6))
            return parcel_handler_.get_thread_pool(name);
        if (0 == std::strncmp(name, "timer", 5))
            return &timer_pool_;
        if (0 == std::strncmp(name, "main", 4)) //-V112
            return &main_pool_;

        return 0;
    }

    /// Register an external OS-thread with HPX
    template <typename SchedulingPolicy, typename NotificationPolicy>
    bool runtime_impl<SchedulingPolicy, NotificationPolicy>::
        register_thread(char const* name, std::size_t num, bool service_thread)
    {
        if (NULL != runtime::thread_name_.get())
            return false;       // already registered

        std::string thread_name(name);
        thread_name += "-thread";

        init_tss(thread_name.c_str(), num, 0, service_thread);

        return true;
    }

    /// Unregister an external OS-thread with HPX
    template <typename SchedulingPolicy, typename NotificationPolicy>
    bool runtime_impl<SchedulingPolicy, NotificationPolicy>::
        unregister_thread()
    {
        if (NULL == runtime::thread_name_.get())
            return false;       // never registered

        deinit_tss();
        return true;
    }
}

///////////////////////////////////////////////////////////////////////////////
/// explicit template instantiation for the thread manager of our choice
#if defined(HPX_LOCAL_SCHEDULER)
#include <hpx/runtime/threads/policies/local_queue_scheduler.hpp>
template class HPX_EXPORT hpx::runtime_impl<
    hpx::threads::policies::local_queue_scheduler<>,
    hpx::threads::policies::callback_notifier>;
#endif

#if defined(HPX_STATIC_PRIORITY_SCHEDULER)
#include <hpx/runtime/threads/policies/static_priority_queue_scheduler.hpp>
template class HPX_EXPORT hpx::runtime_impl<
    hpx::threads::policies::static_priority_queue_scheduler<>,
    hpx::threads::policies::callback_notifier>;
#endif

#include <hpx/runtime/threads/policies/local_priority_queue_scheduler.hpp>
template class HPX_EXPORT hpx::runtime_impl<
    hpx::threads::policies::local_priority_queue_scheduler<>,
    hpx::threads::policies::callback_notifier>;

#if defined(HPX_ABP_SCHEDULER)
template class HPX_EXPORT hpx::runtime_impl<
    hpx::threads::policies::abp_fifo_priority_queue_scheduler,
    hpx::threads::policies::callback_notifier>;
#endif

#if defined(HPX_HIERARCHY_SCHEDULER)
#include <hpx/runtime/threads/policies/hierarchy_scheduler.hpp>
template class HPX_EXPORT hpx::runtime_impl<
    hpx::threads::policies::hierarchy_scheduler<>,
    hpx::threads::policies::callback_notifier>;
#endif

#if defined(HPX_PERIODIC_PRIORITY_SCHEDULER)
#include <hpx/runtime/threads/policies/periodic_priority_queue_scheduler.hpp>
template class HPX_EXPORT hpx::runtime_impl<
    hpx::threads::policies::periodic_priority_queue_scheduler<>,
    hpx::threads::policies::callback_notifier>;
#endif

