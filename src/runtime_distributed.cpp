//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#include <hpx/assert.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/async_distributed/applier/applier.hpp>
#include <hpx/async_distributed/apply.hpp>
#include <hpx/collectives/barrier.hpp>
#include <hpx/collectives/detail/barrier_node.hpp>
#include <hpx/collectives/latch.hpp>
#include <hpx/command_line_handling/command_line_handling.hpp>
#include <hpx/coroutines/coroutine.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/execution_base/this_thread.hpp>
#include <hpx/itt_notify/thread_name.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/functional.hpp>
#include <hpx/modules/logging.hpp>
#include <hpx/modules/static_reinit.hpp>
#include <hpx/modules/threadmanager.hpp>
#include <hpx/modules/topology.hpp>
#include <hpx/performance_counters/counter_creators.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/manage_counter_type.hpp>
#include <hpx/performance_counters/registry.hpp>
#include <hpx/runtime/agas/addressing_service.hpp>
#include <hpx/runtime/agas/big_boot_barrier.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/components/console_error_sink.hpp>
#include <hpx/runtime/components/runtime_support.hpp>
#include <hpx/runtime/components/server/console_error_sink.hpp>
#include <hpx/runtime/components/server/runtime_support.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/runtime/find_localities.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/runtime/parcelset/parcelhandler.hpp>
#include <hpx/runtime/parcelset_fwd.hpp>
#include <hpx/runtime/runtime_fwd.hpp>
#include <hpx/runtime/threads/policies/scheduler_mode.hpp>
#include <hpx/runtime/threads/threadmanager_counters.hpp>
#include <hpx/runtime_configuration/runtime_configuration.hpp>
#include <hpx/runtime_distributed.hpp>
#include <hpx/runtime_local/config_entry.hpp>
#include <hpx/runtime_local/custom_exception_info.hpp>
#include <hpx/runtime_local/debugging.hpp>
#include <hpx/runtime_local/runtime_local.hpp>
#include <hpx/runtime_local/shutdown_function.hpp>
#include <hpx/runtime_local/startup_function.hpp>
#include <hpx/runtime_local/thread_hooks.hpp>
#include <hpx/runtime_local/thread_mapper.hpp>
#include <hpx/state.hpp>
#include <hpx/thread_support/set_thread_name.hpp>
#include <hpx/threading_base/external_timer.hpp>
#include <hpx/timing/high_resolution_clock.hpp>
#include <hpx/util/from_string.hpp>
#include <hpx/util/query_counters.hpp>
#include <hpx/version.hpp>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN64) && defined(_DEBUG) &&                                      \
    !defined(HPX_HAVE_FIBER_BASED_COROUTINES)
#include <io.h>
#endif

///////////////////////////////////////////////////////////////////////////////
static void garbage_collect_non_blocking()
{
    hpx::agas::garbage_collect_non_blocking();
}
static void garbage_collect()
{
    hpx::agas::garbage_collect();
}

///////////////////////////////////////////////////////////////////////////////
namespace hpx {

    ///////////////////////////////////////////////////////////////////////////////
    // Install performance counter startup functions for core subsystems.
    static void register_counter_types()
    {
        naming::get_agas_client().register_counter_types();
        lbt_ << "(2nd stage) pre_main: registered AGAS client-side "
                "performance counter types";

        get_runtime_distributed().register_counter_types();
        lbt_ << "(2nd stage) pre_main: registered runtime performance "
                "counter types";

        threads::register_counter_types(threads::get_thread_manager());
        lbt_ << "(2nd stage) pre_main: registered thread-manager performance "
                "counter types";

#if defined(HPX_HAVE_NETWORKING)
        applier::get_applier().get_parcel_handler().register_counter_types();
        lbt_ << "(2nd stage) pre_main: registered parcelset performance "
                "counter types";
#endif
    }

    ///////////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_NETWORKING)
    // There is no need to protect these global from thread concurrent access
    // as they are access during early startup only.
    std::vector<hpx::util::tuple<char const*, char const*>>&
    get_message_handler_registrations()
    {
        static std::vector<hpx::util::tuple<char const*, char const*>>
            message_handler_registrations;
        return message_handler_registrations;
    }

    static void register_message_handlers()
    {
        runtime_distributed& rtd = get_runtime_distributed();
        for (auto const& t : get_message_handler_registrations())
        {
            error_code ec(lightweight);
            rtd.register_message_handler(util::get<0>(t), util::get<1>(t), ec);
        }
        lbt_ << "(3rd stage) pre_main: registered message handlers";
    }
#endif

    ///////////////////////////////////////////////////////////////////////////////
    // Implements second and third stage bootstrapping.
    int pre_main(runtime_mode mode)
    {
        // Register pre-shutdown and shutdown functions to flush pending
        // reference counting operations.
        register_pre_shutdown_function(&::garbage_collect_non_blocking);
        register_shutdown_function(&::garbage_collect);

        using components::stubs::runtime_support;

        naming::resolver_client& agas_client = naming::get_agas_client();
        runtime& rt = get_runtime();

        int exit_code = 0;
        if (runtime_mode::connect == mode)
        {
            lbt_ << "(2nd stage) pre_main: locality is in connect mode, "
                    "skipping 2nd and 3rd stage startup synchronization";
            lbt_ << "(2nd stage) pre_main: addressing services enabled";

            // Load components, so that we can use the barrier LCO.
            exit_code = runtime_support::load_components(find_here());
            lbt_ << "(2nd stage) pre_main: loaded components"
                 << (exit_code ? ", application exit has been requested" : "");

            // Work on registration requests for message handler plugins
#if defined(HPX_HAVE_NETWORKING)
            register_message_handlers();
#endif

            // Register all counter types before the startup functions are being
            // executed.
            register_counter_types();

            rt.set_state(state_pre_startup);
            runtime_support::call_startup_functions(find_here(), true);
            lbt_ << "(3rd stage) pre_main: ran pre-startup functions";

            rt.set_state(state_startup);
            runtime_support::call_startup_functions(find_here(), false);
            lbt_ << "(4th stage) pre_main: ran startup functions";
        }
        else
        {
            lbt_ << "(2nd stage) pre_main: addressing services enabled";

            // Load components, so that we can use the barrier LCO.
            exit_code = runtime_support::load_components(find_here());
            lbt_ << "(2nd stage) pre_main: loaded components"
                 << (exit_code ? ", application exit has been requested" : "");

            // {{{ Second and third stage barrier creation.
            if (agas_client.is_bootstrap())
            {
                naming::gid_type console_;
                if (HPX_UNLIKELY(!agas_client.get_console_locality(console_)))
                {
                    HPX_THROW_EXCEPTION(network_error, "pre_main",
                        "no console locality registered");
                }

                lbt_ << "(2nd stage) pre_main: creating 2nd and 3rd stage boot "
                        "barriers";
            }
            else    // Hosted.
            {
                lbt_ << "(2nd stage) pre_main: finding 2nd and 3rd stage boot "
                        "barriers";
            }
            // }}}

            // create our global barrier...
            hpx::lcos::barrier::get_global_barrier() =
                hpx::lcos::barrier::create_global_barrier();

            // Second stage bootstrap synchronizes component loading across all
            // localities, ensuring that the component namespace tables are fully
            // populated before user code is executed.
            lcos::barrier::synchronize();
            lbt_ << "(2nd stage) pre_main: passed 2nd stage boot barrier";

            // Work on registration requests for message handler plugins
#if defined(HPX_HAVE_NETWORKING)
            register_message_handlers();
#endif

            // Register all counter types before the startup functions are being
            // executed.
            register_counter_types();

            // Second stage bootstrap synchronizes performance counter loading
            // across all localities.
            lcos::barrier::synchronize();
            lbt_ << "(3rd stage) pre_main: passed 3rd stage boot barrier";

            runtime_support::call_startup_functions(find_here(), true);
            lbt_ << "(3rd stage) pre_main: ran pre-startup functions";

            // Third stage separates pre-startup and startup function phase.
            lcos::barrier::synchronize();
            lbt_ << "(4th stage) pre_main: passed 4th stage boot barrier";

            runtime_support::call_startup_functions(find_here(), false);
            lbt_ << "(4th stage) pre_main: ran startup functions";

            // Forth stage bootstrap synchronizes startup functions across all
            // localities. This is done after component loading to guarantee that
            // all user code, including startup functions, are only run after the
            // component tables are populated.
            lcos::barrier::synchronize();
            lbt_ << "(5th stage) pre_main: passed 4th stage boot barrier";
        }

        // Enable logging. Even if we terminate at this point we will see all
        // pending log messages so far.
        components::activate_logging();
        lbt_ << "(last stage) pre_main: activated logging";

        // Any error in post-command line handling or any explicit --exit command
        // line option will cause the application to terminate at this point.
        if (exit_code)
        {
            // If load_components returns false, shutdown the system. This
            // essentially only happens if the command line contained --exit.
            runtime_support::shutdown_all(
                naming::get_id_from_locality_id(HPX_AGAS_BOOTSTRAP_PREFIX),
                -1.0);
            return exit_code;
        }

        return 0;
    }

}    // namespace hpx

namespace hpx {
    namespace detail {
        naming::gid_type get_next_id(std::size_t count)
        {
            if (nullptr == get_runtime_ptr())
                return naming::invalid_gid;

            return get_runtime_distributed().get_next_id(count);
        }

        ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_NETWORKING)
        void dijkstra_make_black()
        {
            get_runtime_support_ptr()->dijkstra_make_black();
        }
#endif

#if defined(HPX_HAVE_BACKGROUND_THREAD_COUNTERS) &&                            \
    defined(HPX_HAVE_THREAD_IDLE_RATES)
        bool network_background_callback(std::size_t num_thread,
            std::int64_t& background_work_exec_time_send,
            std::int64_t& background_work_exec_time_receive)
        {
            bool result = false;

#if defined(HPX_HAVE_NETWORKING)
            // count background work duration
            {
                threads::background_work_duration_counter bg_send_duration(
                    background_work_exec_time_send);
                threads::background_exec_time_wrapper bg_exec_time(
                    bg_send_duration);

                if (hpx::parcelset::do_background_work(
                        num_thread, parcelset::parcelport_background_mode_send))
                {
                    result = true;
                }
            }

            {
                threads::background_work_duration_counter bg_receive_duration(
                    background_work_exec_time_receive);
                threads::background_exec_time_wrapper bg_exec_time(
                    bg_receive_duration);

                if (hpx::parcelset::do_background_work(num_thread,
                        parcelset::parcelport_background_mode_receive))
                {
                    result = true;
                }
            }
#endif

            if (0 == num_thread)
                hpx::agas::garbage_collect_non_blocking();
            return result;
        }
#else
        bool network_background_callback(std::size_t num_thread)
        {
            bool result = false;

#if defined(HPX_HAVE_NETWORKING)
            if (hpx::parcelset::do_background_work(
                    num_thread, parcelset::parcelport_background_mode_all))
            {
                result = true;
            }
#endif

            if (0 == num_thread)
                hpx::agas::garbage_collect_non_blocking();
            return result;
        }
#endif
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    components::server::runtime_support* get_runtime_support_ptr()
    {
        return reinterpret_cast<components::server::runtime_support*>(
            get_runtime_distributed().get_runtime_support_lva());
    }

    ///////////////////////////////////////////////////////////////////////////
    runtime_distributed::runtime_distributed(util::runtime_configuration& rtcfg)
      : runtime(rtcfg,
            runtime_distributed::get_notification_policy(
                "worker-thread", runtime_local::os_thread_type::worker_thread),
            notification_policy_type{},
#ifdef HPX_HAVE_IO_POOL
            runtime_distributed::get_notification_policy(
                "io-thread", runtime_local::os_thread_type::io_thread),
#endif
#ifdef HPX_HAVE_TIMER_POOL
            runtime_distributed::get_notification_policy(
                "timer-thread", runtime_local::os_thread_type::timer_thread),
#endif
#ifdef HPX_HAVE_NETWORKING
            &detail::network_background_callback,
#endif
            false)
      , mode_(rtcfg.mode_)
#if defined(HPX_HAVE_NETWORKING)
      , parcel_handler_notifier_(runtime_distributed::get_notification_policy(
            "parcel-thread", runtime_local::os_thread_type::parcel_thread))
      , parcel_handler_(rtcfg, thread_manager_.get(), parcel_handler_notifier_)
      , agas_client_(ini_, rtcfg.mode_)
      , applier_(parcel_handler_, *thread_manager_)
#else
      , agas_client_(ini_, rtcfg.mode_)
      , applier_(*thread_manager_)
#endif
      , runtime_support_(new components::server::runtime_support(ini_))
    {
        // This needs to happen first
        runtime::init();

        runtime_distributed*& runtime_distributed_ =
            get_runtime_distributed_ptr();
        if (nullptr == runtime_distributed_)
        {
            HPX_ASSERT(nullptr == threads::thread_self::get_self());

            runtime_distributed_ = this;
        }

        LPROGRESS_;

        counters_ = std::make_shared<performance_counters::registry>();

#if defined(HPX_HAVE_NETWORKING)
        agas_client_.bootstrap(parcel_handler_, ini_);
#else
        agas_client_.bootstrap(ini_);
#endif

        components::server::get_error_dispatcher().set_error_sink(
            &runtime_distributed::default_errorsink);

        // now, launch AGAS and register all nodes, launch all other components
#if defined(HPX_HAVE_NETWORKING)
        agas_client_.initialize(
            parcel_handler_, std::uint64_t(runtime_support_.get()));
        parcel_handler_.initialize(agas_client_, &applier_);
#else
        agas_client_.initialize(std::uint64_t(runtime_support_.get()));
#endif
        applier_.initialize(std::uint64_t(runtime_support_.get()));
    }

    ///////////////////////////////////////////////////////////////////////////
    runtime_distributed::~runtime_distributed()
    {
        LRT_(debug) << "~runtime_distributed(entering)";

        runtime_support_->delete_function_lists();

        // stop all services
#if defined(HPX_HAVE_NETWORKING)
        parcel_handler_.stop();    // stops parcel pools as well
#endif
        // unload libraries
        runtime_support_->tidy();

        LRT_(debug) << "~runtime_distributed(finished)";

        LPROGRESS_;
    }

    threads::thread_result_type runtime_distributed::run_helper(
        util::function_nonser<runtime::hpx_main_function_type> const& func,
        int& result)
    {
        bool caught_exception = false;
        try
        {
            lbt_ << "(2nd stage) runtime_distributed::run_helper: launching "
                    "pre_main";

            // Change our thread description, as we're about to call pre_main
            threads::set_thread_description(threads::get_self_id(), "pre_main");

            // Finish the bootstrap
            result = hpx::pre_main(mode_);
            if (result)
            {
                lbt_ << "runtime_distributed::run_helper: bootstrap "
                        "aborted, bailing out";
                return threads::thread_result_type(
                    threads::terminated, threads::invalid_thread_id);
            }

            lbt_ << "(4th stage) runtime_distributed::run_helper: bootstrap "
                    "complete";
            set_state(state_running);

#if defined(HPX_HAVE_NETWORKING)
            parcel_handler_.enable_alternative_parcelports();
#endif

            // reset all counters right before running main, if requested
            if (get_config_entry("hpx.print_counter.startup", "0") == "1")
            {
                bool reset = false;
                if (get_config_entry("hpx.print_counter.reset", "0") == "1")
                    reset = true;

                error_code ec(lightweight);    // ignore errors
                evaluate_active_counters(reset, "startup", ec);
            }

            // Connect back to given latch if specified
            std::string connect_back_to(
                get_config_entry("hpx.on_startup.wait_on_latch", ""));
            if (!connect_back_to.empty())
            {
                lbt_ << "(5th stage) runtime::run_helper: about to "
                        "synchronize with latch: "
                     << connect_back_to;

                // inform launching process that this locality is up and running
                hpx::lcos::latch l;
                l.connect_to(connect_back_to);
                l.count_down_and_wait();

                lbt_ << "(5th stage) runtime::run_helper: "
                        "synchronized with latch: "
                     << connect_back_to;
            }
        }
        catch (...)
        {
            // make sure exceptions thrown in hpx_main don't escape
            // unnoticed
            {
                std::lock_guard<std::mutex> l(mtx_);
                exception_ = std::current_exception();
            }
            result = -1;
            caught_exception = true;
        }

        if (caught_exception)
        {
            HPX_ASSERT(exception_);
            report_error(exception_, false);
            finalize(-1.0);    // make sure the application exits
        }

        return runtime::run_helper(func, result, false);
    }

    int runtime_distributed::start(
        util::function_nonser<hpx_main_function_type> const& func,
        bool blocking)
    {
#if defined(_WIN64) && defined(_DEBUG) &&                                      \
    !defined(HPX_HAVE_FIBER_BASED_COROUTINES)
        // needs to be called to avoid problems at system startup
        // see: http://connect.microsoft.com/VisualStudio/feedback/ViewFeedback.aspx?FeedbackID=100319
        _isatty(0);
#endif
        // {{{ early startup code - local

        // initialize instrumentation system
#ifdef HPX_HAVE_APEX
        util::external_timer::init(
            nullptr, hpx::get_locality_id(), hpx::get_initial_num_localities());
#endif

        LRT_(info) << "cmd_line: " << get_config().get_cmd_line();

        lbt_ << "(1st stage) runtime_distributed::start: booting locality "
             << here();

        // Register this thread with the runtime system to allow calling
        // certain HPX functionality from the main thread. Also calls
        // registered startup callbacks.
        init_tss_helper("main-thread",
            runtime_local::os_thread_type::main_thread, 0, 0, "", "", false);

        // start runtime_support services
        runtime_support_->run();
        lbt_ << "(1st stage) runtime_distributed::start: started "
                "runtime_support component";

#ifdef HPX_HAVE_IO_POOL
        // start the io pool
        io_pool_.run(false);
        lbt_ << "(1st stage) runtime_distributed::start: started the "
                "application "
                "I/O service pool";
#endif
        // start the thread manager
        thread_manager_->run();
        lbt_ << "(1st stage) runtime_distributed::start: started threadmanager";
        // }}}

        // invoke the AGAS v2 notifications
#if defined(HPX_HAVE_NETWORKING)
        agas::get_big_boot_barrier().trigger();
#endif

        // {{{ launch main
        // register the given main function with the thread manager
        lbt_ << "(1st stage) runtime_distributed::start: launching run_helper "
                "HPX thread";

        threads::thread_init_data data(
            util::bind(&runtime_distributed::run_helper, this, func,
                std::ref(result_)),
            "run_helper", threads::thread_priority_normal,
            threads::thread_schedule_hint(0), threads::thread_stacksize_large);

        this->runtime::starting();
        threads::thread_id_type id = threads::invalid_thread_id;
        thread_manager_->register_thread(data, id);

        // }}}

        // block if required
        if (blocking)
        {
            return wait();    // wait for the shutdown_action to be executed
        }
        else
        {
            // wait for at least state_running
            util::yield_while([this]() { return get_state() < state_running; },
                "runtime_impl::start");
        }

        return 0;    // return zero as we don't know the outcome of hpx_main yet
    }

    int runtime_distributed::start(bool blocking)
    {
        util::function_nonser<hpx_main_function_type> empty_main;
        return start(empty_main, blocking);
    }

    ///////////////////////////////////////////////////////////////////////////
    std::string locality_prefix(util::runtime_configuration const& cfg)
    {
        std::string localities = cfg.get_entry("hpx.localities", "1");
        std::size_t num_localities =
            util::from_string<std::size_t>(localities, 1);
        if (num_localities > 1)
        {
            std::string locality = cfg.get_entry("hpx.locality", "");
            if (!locality.empty())
            {
                locality = "locality#" + locality;
            }
            return locality;
        }
        return "";
    }

    ///////////////////////////////////////////////////////////////////////////
    void runtime_distributed::wait_helper(
        std::mutex& mtx, std::condition_variable& cond, bool& running)
    {
        // signal successful initialization
        {
            std::lock_guard<std::mutex> lk(mtx);
            running = true;
            cond.notify_all();
        }

        // prefix thread name with locality number, if needed
        std::string locality = locality_prefix(get_config());

        // register this thread with any possibly active Intel tool
        std::string thread_name(locality + "main-thread#wait_helper");
        HPX_ITT_THREAD_SET_NAME(thread_name.c_str());

        // set thread name as shown in Visual Studio
        util::set_thread_name(thread_name.c_str());

#if defined(HPX_HAVE_APEX)
        // not registering helper threads - for now
        //util::external_timer::register_thread(thread_name.c_str());
#endif

        // wait for termination
        runtime_support_->wait();

        // stop main thread pool
        main_pool_.stop();
    }

    int runtime_distributed::wait()
    {
        LRT_(info) << "runtime_distributed: about to enter wait state";

        // start the wait_helper in a separate thread
        std::mutex mtx;
        std::condition_variable cond;
        bool running = false;

        std::thread t(util::bind(&runtime_distributed::wait_helper, this,
            std::ref(mtx), std::ref(cond), std::ref(running)));

        // wait for the thread to run
        {
            std::unique_lock<std::mutex> lk(mtx);
            while (!running)    // -V776 // -V1044
                cond.wait(lk);
        }

        // use main thread to drive main thread pool
        main_pool_.thread_run(0);

        // block main thread
        t.join();

        LRT_(info) << "runtime_distributed: exiting wait state";
        return result_;
    }

    ///////////////////////////////////////////////////////////////////////////
    // First half of termination process: stop thread manager,
    // schedule a task managed by timer_pool to initiate second part
    void runtime_distributed::stop(bool blocking)
    {
        LRT_(warning) << "runtime_distributed: about to stop services";

        // flush all parcel buffers, stop buffering parcels at this point
        //parcel_handler_.do_background_work(true, parcelport_background_mode_all);

        // execute all on_exit functions whenever the first thread calls this
        this->runtime::stopping();

        // stop runtime_distributed services (threads)
        thread_manager_->stop(false);    // just initiate shutdown

#ifdef HPX_HAVE_APEX
        util::external_timer::finalize();
#endif

        if (threads::get_self_ptr())
        {
            // schedule task on separate thread to execute stop_helper() below
            // this is necessary as this function (stop()) might have been called
            // from a HPX thread, so it would deadlock by waiting for the thread
            // manager
            std::mutex mtx;
            std::condition_variable cond;
            std::unique_lock<std::mutex> l(mtx);

            std::thread t(util::bind(&runtime_distributed::stop_helper, this,
                blocking, std::ref(cond), std::ref(mtx)));
            cond.wait(l);

            t.join();
        }
        else
        {
            runtime_support_->stopped();    // re-activate shutdown HPX-thread
            thread_manager_->stop(blocking);    // wait for thread manager

            // this disables all logging from the main thread
            deinit_tss_helper("main-thread", 0);

            LRT_(info) << "runtime_distributed: stopped all services";
        }

        // stop the rest of the system
#if defined(HPX_HAVE_NETWORKING)
        parcel_handler_.stop(blocking);    // stops parcel pools as well
#endif
#ifdef HPX_HAVE_TIMER_POOL
        LTM_(info) << "stop: stopping timer pool";
        timer_pool_.stop();    // stop timer pool as well
        if (blocking)
        {
            timer_pool_.join();
            timer_pool_.clear();
        }
#endif
#ifdef HPX_HAVE_IO_POOL
        io_pool_.stop();    // stops io_pool_ as well
#endif
        // deinit_tss();
    }

    int runtime_distributed::finalize(double shutdown_timeout)
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        //   tell main locality to start application exit, duplicated requests
        // will be ignored
        apply<components::server::runtime_support::shutdown_all_action>(
            hpx::find_root_locality(), shutdown_timeout);
#endif

        return 0;
    }

    // Second step in termination: shut down all services.
    // This gets executed as a task in the timer_pool io_service and not as
    // a HPX thread!
    void runtime_distributed::stop_helper(
        bool blocking, std::condition_variable& cond, std::mutex& mtx)
    {
        // wait for thread manager to exit
        runtime_support_->stopped();        // re-activate shutdown HPX-thread
        thread_manager_->stop(blocking);    // wait for thread manager

        // this disables all logging from the main thread
        deinit_tss_helper("main-thread", 0);

        LRT_(info) << "runtime_distributed: stopped all services";

        std::lock_guard<std::mutex> l(mtx);
        cond.notify_all();    // we're done now
    }

    int runtime_distributed::suspend()
    {
#if defined(HPX_HAVE_NETWORKING)
        std::uint32_t initial_num_localities = get_initial_num_localities();
        if (initial_num_localities > 1)
        {
            HPX_THROW_EXCEPTION(invalid_status, "runtime_distributed::suspend",
                "Can only suspend runtime when number of localities is 1");
            return -1;
        }
#endif

        return runtime::suspend();
    }

    int runtime_distributed::resume()
    {
#if defined(HPX_HAVE_NETWORKING)
        std::uint32_t initial_num_localities = get_initial_num_localities();
        if (initial_num_localities > 1)
        {
            HPX_THROW_EXCEPTION(invalid_status, "runtime_distributed::resume",
                "Can only suspend runtime when number of localities is 1");
            return -1;
        }
#endif

        return runtime::resume();
    }

    ///////////////////////////////////////////////////////////////////////////
    bool runtime_distributed::report_error(
        std::size_t num_thread, std::exception_ptr const& e, bool terminate_all)
    {
        // call thread-specific user-supplied on_error handler
        bool report_exception = true;
        if (on_error_func_)
        {
            report_exception = on_error_func_(num_thread, e);
        }

        // Early and late exceptions, errors outside of HPX-threads
        if (!threads::get_self_ptr() ||
            !threads::threadmanager_is(state_running))
        {
            // report the error to the local console
            if (report_exception)
            {
                detail::report_exception_and_continue(e);
            }

            // store the exception to be able to rethrow it later
            {
                std::lock_guard<std::mutex> l(mtx_);
                exception_ = e;
            }

            lcos::barrier::get_global_barrier().detach();

            // initiate stopping the runtime system
            runtime_support_->notify_waiting_main();
            stop(false);

            return report_exception;
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
            if (agas_client_.get_local_locality() != console_id)
            {
                components::console_error_sink(
                    naming::id_type(console_id, naming::id_type::unmanaged), e);
            }
        }

        if (terminate_all)
        {
            components::stubs::runtime_support::terminate_all(
                naming::get_id_from_locality_id(HPX_AGAS_BOOTSTRAP_PREFIX));
        }

        return report_exception;
    }

    bool runtime_distributed::report_error(
        std::exception_ptr const& e, bool terminate_all)
    {
        return report_error(hpx::get_worker_thread_num(), e, terminate_all);
    }

    ///////////////////////////////////////////////////////////////////////////
    int runtime_distributed::run(
        util::function_nonser<hpx_main_function_type> const& func)
    {
        // start the main thread function
        start(func);

        // now wait for everything to finish
        wait();
        stop();

#if defined(HPX_HAVE_NETWORKING)
        parcel_handler_.stop();    // stops parcelport for sure
#endif

        rethrow_exception();
        return result_;
    }

    ///////////////////////////////////////////////////////////////////////////
    int runtime_distributed::run()
    {
        // start the main thread function
        start();

        // now wait for everything to finish
        int result = wait();
        stop();

#if defined(HPX_HAVE_NETWORKING)
        parcel_handler_.stop();    // stops parcelport for sure
#endif

        rethrow_exception();
        return result;
    }

    bool runtime_distributed::is_networking_enabled()
    {
#if defined(HPX_HAVE_NETWORKING)
        return get_config().enable_networking();
#else
        return false;
#endif
    }

    performance_counters::registry& runtime_distributed::get_counter_registry()
    {
        return *counters_;
    }

    performance_counters::registry const&
    runtime_distributed::get_counter_registry() const
    {
        return *counters_;
    }

    ///////////////////////////////////////////////////////////////////////////
    void runtime_distributed::register_query_counters(
        std::shared_ptr<util::query_counters> const& active_counters)
    {
        active_counters_ = active_counters;
    }

    void runtime_distributed::start_active_counters(error_code& ec)
    {
        if (active_counters_.get())
            active_counters_->start_counters(ec);
    }

    void runtime_distributed::stop_active_counters(error_code& ec)
    {
        if (active_counters_.get())
            active_counters_->stop_counters(ec);
    }

    void runtime_distributed::reset_active_counters(error_code& ec)
    {
        if (active_counters_.get())
            active_counters_->reset_counters(ec);
    }

    void runtime_distributed::reinit_active_counters(bool reset, error_code& ec)
    {
        if (active_counters_.get())
            active_counters_->reinit_counters(reset, ec);
    }

    void runtime_distributed::evaluate_active_counters(
        bool reset, char const* description, error_code& ec)
    {
        if (active_counters_.get())
            active_counters_->evaluate_counters(reset, description, true, ec);
    }

    void runtime_distributed::stop_evaluating_counters(bool terminate)
    {
        if (active_counters_.get())
            active_counters_->stop_evaluating_counters(terminate);
    }

    naming::resolver_client& runtime_distributed::get_agas_client()
    {
        return agas_client_;
    }

#if defined(HPX_HAVE_NETWORKING)
    parcelset::parcelhandler const& runtime_distributed::get_parcel_handler()
        const
    {
        return parcel_handler_;
    }

    parcelset::parcelhandler& runtime_distributed::get_parcel_handler()
    {
        return parcel_handler_;
    }
#endif

    hpx::threads::threadmanager& runtime_distributed::get_thread_manager()
    {
        return *thread_manager_;
    }

    applier::applier& runtime_distributed::get_applier()
    {
        return applier_;
    }

#if defined(HPX_HAVE_NETWORKING)
    parcelset::endpoints_type const& runtime_distributed::endpoints() const
    {
        return parcel_handler_.endpoints();
    }
#endif

    std::string runtime_distributed::here() const
    {
#if defined(HPX_HAVE_NETWORKING)
        std::ostringstream strm;
        strm << endpoints();
        return strm.str();
#else
        return "console";
#endif
    }

    std::uint64_t runtime_distributed::get_runtime_support_lva() const
    {
        return reinterpret_cast<std::uint64_t>(runtime_support_.get());
    }

    naming::gid_type get_next_id(std::size_t count = 1);

    util::unique_id_ranges& runtime_distributed::get_id_pool()
    {
        return id_pool_;
    }

    /// \brief Register all performance counter types related to this runtime
    ///        instance
    void runtime_distributed::register_counter_types()
    {
        performance_counters::generic_counter_type_data
            statistic_counter_types[] =
        {    // averaging counter
            {"/statistics/average", performance_counters::counter_aggregating,
                "returns the averaged value of its base counter over "
                "an arbitrary time line; pass required base counter as the "
                "instance "
                "name: /statistics{<base_counter_name>}/average",
                HPX_PERFORMANCE_COUNTER_V1,
                &performance_counters::detail::statistics_counter_creator,
                &performance_counters::default_counter_discoverer, ""},

            // stddev counter
            {"/statistics/stddev", performance_counters::counter_aggregating,
                "returns the standard deviation value of its base counter "
                "over "
                "an arbitrary time line; pass required base counter as the "
                "instance "
                "name: /statistics{<base_counter_name>}/stddev",
                HPX_PERFORMANCE_COUNTER_V1,
                &performance_counters::detail::statistics_counter_creator,
                &performance_counters::default_counter_discoverer, ""},

            // rolling_averaging counter
            {"/statistics/rolling_average",
                performance_counters::counter_aggregating,
                "returns the rolling average value of its base counter "
                "over "
                "an arbitrary time line; pass required base counter as the "
                "instance "
                "name: /statistics{<base_counter_name>}/rolling_averaging",
                HPX_PERFORMANCE_COUNTER_V1,
                &performance_counters::detail::statistics_counter_creator,
                &performance_counters::default_counter_discoverer, ""},

            // rolling stddev counter
            {"/statistics/rolling_stddev",
                performance_counters::counter_aggregating,
                "returns the rolling standard deviation value of its base "
                "counter over "
                "an arbitrary time line; pass required base counter as the "
                "instance "
                "name: /statistics{<base_counter_name>}/rolling_stddev",
                HPX_PERFORMANCE_COUNTER_V1,
                &performance_counters::detail::statistics_counter_creator,
                &performance_counters::default_counter_discoverer, ""},

            // median counter
            {"/statistics/median", performance_counters::counter_aggregating,
                "returns the median value of its base counter over "
                "an arbitrary time line; pass required base counter as the "
                "instance "
                "name: /statistics{<base_counter_name>}/median",
                HPX_PERFORMANCE_COUNTER_V1,
                &performance_counters::detail::statistics_counter_creator,
                &performance_counters::default_counter_discoverer, ""},

            // max counter
            {"/statistics/max", performance_counters::counter_aggregating,
                "returns the maximum value of its base counter over "
                "an arbitrary time line; pass required base counter as the "
                "instance "
                "name: /statistics{<base_counter_name>}/max",
                HPX_PERFORMANCE_COUNTER_V1,
                &performance_counters::detail::statistics_counter_creator,
                &performance_counters::default_counter_discoverer, ""},

            // min counter
            {"/statistics/min", performance_counters::counter_aggregating,
                "returns the minimum value of its base counter over "
                "an arbitrary time line; pass required base counter as the "
                "instance "
                "name: /statistics{<base_counter_name>}/min",
                HPX_PERFORMANCE_COUNTER_V1,
                &performance_counters::detail::statistics_counter_creator,
                &performance_counters::default_counter_discoverer, ""},

            // rolling max counter
            {"/statistics/rolling_max",
                performance_counters::counter_aggregating,
                "returns the rolling maximum value of its base counter "
                "over "
                "an arbitrary time line; pass required base counter as the "
                "instance "
                "name: /statistics{<base_counter_name>}/rolling_max",
                HPX_PERFORMANCE_COUNTER_V1,
                &performance_counters::detail::statistics_counter_creator,
                &performance_counters::default_counter_discoverer, ""},

            // rolling min counter
            {"/statistics/rolling_min",
                performance_counters::counter_aggregating,
                "returns the rolling minimum value of its base counter "
                "over "
                "an arbitrary time line; pass required base counter as the "
                "instance "
                "name: /statistics{<base_counter_name>}/rolling_min",
                HPX_PERFORMANCE_COUNTER_V1,
                &performance_counters::detail::statistics_counter_creator,
                &performance_counters::default_counter_discoverer, ""},

            // uptime counters
            {
                "/runtime/uptime", performance_counters::counter_elapsed_time,
                "returns the up time of the runtime instance for the "
                "referenced "
                "locality",
                HPX_PERFORMANCE_COUNTER_V1,
                &performance_counters::detail::uptime_counter_creator,
                &performance_counters::locality_counter_discoverer,
                "s"    // unit of measure is seconds
            },

            // component instance counters
            {"/runtime/count/component", performance_counters::counter_raw,
                "returns the number of component instances currently alive "
                "on "
                "this locality (the component type has to be specified as "
                "the "
                "counter parameter)",
                HPX_PERFORMANCE_COUNTER_V1,
                &performance_counters::detail::
                    component_instance_counter_creator,
                &performance_counters::locality_counter_discoverer, ""},

            // action invocation counters
            {"/runtime/count/action-invocation",
                performance_counters::counter_raw,
                "returns the number of (local) invocations of a specific "
                "action "
                "on this locality (the action type has to be specified as "
                "the "
                "counter parameter)",
                HPX_PERFORMANCE_COUNTER_V1,
                &performance_counters::local_action_invocation_counter_creator,
                &performance_counters::
                    local_action_invocation_counter_discoverer,
                ""},

#if defined(HPX_HAVE_NETWORKING)
            {"/runtime/count/remote-action-invocation",
                performance_counters::counter_raw,
                "returns the number of (remote) invocations of a specific "
                "action "
                "on this locality (the action type has to be specified as "
                "the "
                "counter parameter)",
                HPX_PERFORMANCE_COUNTER_V1,
                &performance_counters::remote_action_invocation_counter_creator,
                &performance_counters::
                    remote_action_invocation_counter_discoverer,
                ""}
#endif
        };
        performance_counters::install_counter_types(statistic_counter_types,
            sizeof(statistic_counter_types) /
                sizeof(statistic_counter_types[0]));

        performance_counters::generic_counter_type_data
            arithmetic_counter_types[] = {
                // adding counter
                {"/arithmetics/add", performance_counters::counter_aggregating,
                    "returns the sum of the values of the specified base "
                    "counters; "
                    "pass required base counters as the parameters: "
                    "/arithmetics/"
                    "add@<base_counter_name1>,<base_counter_name2>",
                    HPX_PERFORMANCE_COUNTER_V1,
                    &performance_counters::detail::arithmetics_counter_creator,
                    &performance_counters::default_counter_discoverer, ""},
                // minus counter
                {"/arithmetics/subtract",
                    performance_counters::counter_aggregating,
                    "returns the difference of the values of the specified "
                    "base counters; "
                    "pass the required base counters as the parameters: "
                    "/arithmetics/"
                    "subtract@<base_counter_name1>,<base_counter_name2>",
                    HPX_PERFORMANCE_COUNTER_V1,
                    &performance_counters::detail::arithmetics_counter_creator,
                    &performance_counters::default_counter_discoverer, ""},
                // multiply counter
                {"/arithmetics/multiply",
                    performance_counters::counter_aggregating,
                    "returns the product of the values of the specified base "
                    "counters; "
                    "pass the required base counters as the parameters: "
                    "/arithmetics/"
                    "multiply@<base_counter_name1>,<base_counter_name2>",
                    HPX_PERFORMANCE_COUNTER_V1,
                    &performance_counters::detail::arithmetics_counter_creator,
                    &performance_counters::default_counter_discoverer, ""},
                // divide counter
                {"/arithmetics/divide",
                    performance_counters::counter_aggregating,
                    "returns the result of division of the values of the "
                    "specified "
                    "base counters; pass the required base counters as the "
                    "parameters: "
                    "/arithmetics/"
                    "divide@<base_counter_name1>,<base_counter_name2>",
                    HPX_PERFORMANCE_COUNTER_V1,
                    &performance_counters::detail::arithmetics_counter_creator,
                    &performance_counters::default_counter_discoverer, ""},

                // arithmetics mean counter
                {"/arithmetics/mean", performance_counters::counter_aggregating,
                    "returns the average value of all values of the specified "
                    "base counters; pass the required base counters as the "
                    "parameters: "
                    "/arithmetics/"
                    "mean@<base_counter_name1>,<base_counter_name2>",
                    HPX_PERFORMANCE_COUNTER_V1,
                    &performance_counters::detail::
                        arithmetics_counter_extended_creator,
                    &performance_counters::default_counter_discoverer, ""},
                // arithmetics variance counter
                {"/arithmetics/variance",
                    performance_counters::counter_aggregating,
                    "returns the standard deviation of all values of the "
                    "specified "
                    "base counters; pass the required base counters as the "
                    "parameters: "
                    "/arithmetics/"
                    "variance@<base_counter_name1>,<base_counter_name2>",
                    HPX_PERFORMANCE_COUNTER_V1,
                    &performance_counters::detail::
                        arithmetics_counter_extended_creator,
                    &performance_counters::default_counter_discoverer, ""},
                // arithmetics median counter
                {"/arithmetics/median",
                    performance_counters::counter_aggregating,
                    "returns the median of all values of the specified "
                    "base counters; pass the required base counters as the "
                    "parameters: "
                    "/arithmetics/"
                    "median@<base_counter_name1>,<base_counter_name2>",
                    HPX_PERFORMANCE_COUNTER_V1,
                    &performance_counters::detail::
                        arithmetics_counter_extended_creator,
                    &performance_counters::default_counter_discoverer, ""},
                // arithmetics min counter
                {"/arithmetics/min", performance_counters::counter_aggregating,
                    "returns the minimum value of all values of the specified "
                    "base counters; pass the required base counters as the "
                    "parameters: "
                    "/arithmetics/"
                    "min@<base_counter_name1>,<base_counter_name2>",
                    HPX_PERFORMANCE_COUNTER_V1,
                    &performance_counters::detail::
                        arithmetics_counter_extended_creator,
                    &performance_counters::default_counter_discoverer, ""},
                // arithmetics max counter
                {"/arithmetics/max", performance_counters::counter_aggregating,
                    "returns the maximum value of all values of the specified "
                    "base counters; pass the required base counters as the "
                    "parameters: "
                    "/arithmetics/"
                    "max@<base_counter_name1>,<base_counter_name2>",
                    HPX_PERFORMANCE_COUNTER_V1,
                    &performance_counters::detail::
                        arithmetics_counter_extended_creator,
                    &performance_counters::default_counter_discoverer, ""},
                // arithmetics count counter
                {"/arithmetics/count",
                    performance_counters::counter_aggregating,
                    "returns the count value of all values of the specified "
                    "base counters; pass the required base counters as the "
                    "parameters: "
                    "/arithmetics/"
                    "count@<base_counter_name1>,<base_counter_name2>",
                    HPX_PERFORMANCE_COUNTER_V1,
                    &performance_counters::detail::
                        arithmetics_counter_extended_creator,
                    &performance_counters::default_counter_discoverer, ""},
            };
        performance_counters::install_counter_types(arithmetic_counter_types,
            sizeof(arithmetic_counter_types) /
                sizeof(arithmetic_counter_types[0]));
    }

    ///////////////////////////////////////////////////////////////////////////
    void start_active_counters(error_code& ec)
    {
        runtime_distributed* rtd = get_runtime_distributed_ptr();
        if (nullptr != rtd)
        {
            rtd->start_active_counters(ec);
        }
        else
        {
            HPX_THROWS_IF(ec, invalid_status, "start_active_counters",
                "the runtime system is not available at this time");
        }
    }

    void stop_active_counters(error_code& ec)
    {
        runtime_distributed* rtd = get_runtime_distributed_ptr();
        if (nullptr != rtd)
        {
            rtd->stop_active_counters(ec);
        }
        else
        {
            HPX_THROWS_IF(ec, invalid_status, "stop_active_counters",
                "the runtime system is not available at this time");
        }
    }

    void reset_active_counters(error_code& ec)
    {
        runtime_distributed* rtd = get_runtime_distributed_ptr();
        if (nullptr != rtd)
        {
            rtd->reset_active_counters(ec);
        }
        else
        {
            HPX_THROWS_IF(ec, invalid_status, "reset_active_counters",
                "the runtime system is not available at this time");
        }
    }

    void reinit_active_counters(bool reset, error_code& ec)
    {
        runtime_distributed* rtd = get_runtime_distributed_ptr();
        if (nullptr != rtd)
        {
            rtd->reinit_active_counters(reset, ec);
        }
        else
        {
            HPX_THROWS_IF(ec, invalid_status, "reinit_active_counters",
                "the runtime system is not available at this time");
        }
    }

    void evaluate_active_counters(
        bool reset, char const* description, error_code& ec)
    {
        runtime_distributed* rtd = get_runtime_distributed_ptr();
        if (nullptr != rtd)
        {
            rtd->evaluate_active_counters(reset, description, ec);
        }
        else
        {
            HPX_THROWS_IF(ec, invalid_status, "evaluate_active_counters",
                "the runtime system is not available at this time");
        }
    }

    // helper function to stop evaluating counters during shutdown
    void stop_evaluating_counters(bool terminate)
    {
        runtime_distributed* rtd = get_runtime_distributed_ptr();
        if (nullptr != rtd)
            rtd->stop_evaluating_counters(terminate);
    }

    ///////////////////////////////////////////////////////////////////////////
    threads::policies::callback_notifier
    runtime_distributed::get_notification_policy(
        char const* prefix, runtime_local::os_thread_type type)
    {
        typedef bool (runtime_distributed::*report_error_t)(
            std::size_t, std::exception_ptr const&, bool);

        using util::placeholders::_1;
        using util::placeholders::_2;
        using util::placeholders::_3;
        using util::placeholders::_4;

        notification_policy_type notifier;

        notifier.add_on_start_thread_callback(util::bind(
            &runtime_distributed::init_tss_helper, runtime_distributed::This(),
            prefix, type, _1, _2, _3, _4, false));
        notifier.add_on_stop_thread_callback(
            util::bind(&runtime_distributed::deinit_tss_helper,
                runtime_distributed::This(), prefix, _1));
        notifier.set_on_error_callback(util::bind(
            static_cast<report_error_t>(&runtime_distributed::report_error),
            runtime_distributed::This(), _1, _2, true));

        return notifier;
    }

    void runtime_distributed::init_tss_helper(char const* context,
        runtime_local::os_thread_type type, std::size_t local_thread_num,
        std::size_t global_thread_num, char const* pool_name,
        char const* postfix, bool service_thread)
    {
        // prefix thread name with locality number, if needed
        std::string locality = locality_prefix(get_config());

        error_code ec(lightweight);
        return init_tss_ex(locality, context, type, local_thread_num,
            global_thread_num, pool_name, postfix, service_thread, ec);
    }

    void runtime_distributed::init_tss_ex(std::string const& locality,
        char const* context, runtime_local::os_thread_type type,
        std::size_t local_thread_num, std::size_t global_thread_num,
        char const* pool_name, char const* postfix, bool service_thread,
        error_code& ec)
    {
        // initialize our TSS
        runtime::init_tss();
        runtime_distributed*& runtime_distributed_ =
            get_runtime_distributed_ptr();
        if (nullptr == runtime_distributed_)
        {
            HPX_ASSERT(nullptr == threads::thread_self::get_self());

            runtime_distributed_ = this;
        }

        // set the thread's name, if it's not already set
        HPX_ASSERT(detail::thread_name().empty());

        std::string fullname = std::string(locality);
        if (!locality.empty())
            fullname += "/";
        fullname += context;
        if (postfix && *postfix)
            fullname += postfix;
        fullname += "#" + std::to_string(global_thread_num);
        detail::thread_name() = std::move(fullname);

        char const* name = detail::thread_name().c_str();

        // initialize thread mapping for external libraries (i.e. PAPI)
        thread_support_->register_thread(name, type);

        // register this thread with any possibly active Intel tool
        HPX_ITT_THREAD_SET_NAME(name);

        // set thread name as shown in Visual Studio
        util::set_thread_name(name);

#if defined(HPX_HAVE_APEX)
        if (std::strstr(name, "worker") != nullptr)
            util::external_timer::register_thread(name);
#endif

        // call thread-specific user-supplied on_start handler
        if (on_start_func_)
        {
            on_start_func_(
                local_thread_num, global_thread_num, pool_name, context);
        }

        // if this is a service thread, set its service affinity
        if (service_thread)
        {
            // FIXME: We don't set the affinity of the service threads on BG/Q,
            // as this is causing a hang (needs to be investigated)
#if !defined(__bgq__)
            threads::mask_cref_type used_processing_units =
                thread_manager_->get_used_processing_units();

            // --hpx:bind=none  should disable all affinity definitions
            if (threads::any(used_processing_units))
            {
                this->topology_.set_thread_affinity_mask(
                    this->topology_.get_service_affinity_mask(
                        used_processing_units),
                    ec);

                // comment this out for now as on CIrcleCI this is causing unending grief
                //if (ec)
                //{
                //    HPX_THROW_EXCEPTION(kernel_error
                //        , "runtime_distributed::init_tss_ex"
                //        , hpx::util::format(
                //            "failed to set thread affinity mask ("
                //            HPX_CPU_MASK_PREFIX "{:x}) for service thread: {}",
                //            used_processing_units, detail::thread_name()));
                //}
            }
#endif
        }
    }

    void runtime_distributed::deinit_tss_helper(
        char const* context, std::size_t global_thread_num)
    {
        // call thread-specific user-supplied on_stop handler
        if (on_stop_func_)
        {
            on_stop_func_(global_thread_num, global_thread_num, "", context);
        }

        // reset our TSS
        runtime::deinit_tss();
        get_runtime_distributed_ptr() = nullptr;

        // reset PAPI support
        thread_support_->unregister_thread();

        // reset thread local storage
        detail::thread_name().clear();
    }

    naming::gid_type runtime_distributed::get_next_id(std::size_t count)
    {
        return id_pool_.get_id(count);
    }

    void runtime_distributed::add_pre_startup_function(startup_function_type f)
    {
        runtime_support_->add_pre_startup_function(std::move(f));
    }

    void runtime_distributed::add_startup_function(startup_function_type f)
    {
        runtime_support_->add_startup_function(std::move(f));
    }

    void runtime_distributed::add_pre_shutdown_function(
        shutdown_function_type f)
    {
        runtime_support_->add_pre_shutdown_function(std::move(f));
    }

    void runtime_distributed::add_shutdown_function(shutdown_function_type f)
    {
        runtime_support_->add_shutdown_function(std::move(f));
    }

    hpx::util::io_service_pool* runtime_distributed::get_thread_pool(
        char const* name)
    {
        HPX_ASSERT(name != nullptr);
#ifdef HPX_HAVE_IO_POOL
        if (0 == std::strncmp(name, "io", 2))
            return &io_pool_;
#endif
#if defined(HPX_HAVE_NETWORKING)
        if (0 == std::strncmp(name, "parcel", 6))
            return parcel_handler_.get_thread_pool(name);
#endif
#ifdef HPX_HAVE_TIMER_POOL
        if (0 == std::strncmp(name, "timer", 5))
            return &timer_pool_;
#endif
        if (0 == std::strncmp(name, "main", 4))    //-V112
            return &main_pool_;

        HPX_THROW_EXCEPTION(bad_parameter,
            "runtime_distributed::get_thread_pool",
            std::string("unknown thread pool requested: ") + name);
        return nullptr;
    }

    /// Register an external OS-thread with HPX
    bool runtime_distributed::register_thread(char const* name,
        std::size_t global_thread_num, bool service_thread, error_code& ec)
    {
        if (nullptr != get_runtime_ptr())
            return false;    // already registered

        // prefix thread name with locality number, if needed
        std::string locality = locality_prefix(get_config());

        std::string thread_name(name);
        thread_name += "-thread";

        init_tss_ex(locality, thread_name.c_str(),
            runtime_local::os_thread_type::custom_thread, global_thread_num,
            global_thread_num, "", nullptr, service_thread, ec);

        return !ec ? true : false;
    }

#if defined(HPX_HAVE_NETWORKING)
    void runtime_distributed::register_message_handler(
        char const* message_handler_type, char const* action, error_code& ec)
    {
        return runtime_support_->register_message_handler(
            message_handler_type, action, ec);
    }

    parcelset::policies::message_handler*
    runtime_distributed::create_message_handler(
        char const* message_handler_type, char const* action,
        parcelset::parcelport* pp, std::size_t num_messages,
        std::size_t interval, error_code& ec)
    {
        return runtime_support_->create_message_handler(
            message_handler_type, action, pp, num_messages, interval, ec);
    }

    serialization::binary_filter* runtime_distributed::create_binary_filter(
        char const* binary_filter_type, bool compress,
        serialization::binary_filter* next_filter, error_code& ec)
    {
        return runtime_support_->create_binary_filter(
            binary_filter_type, compress, next_filter, ec);
    }
#endif

    std::uint32_t runtime_distributed::get_locality_id(error_code& ec) const
    {
        return agas::get_locality_id(ec);
    }

    std::size_t runtime_distributed::get_num_worker_threads() const
    {
        error_code ec(lightweight);
        return static_cast<std::size_t>(
            agas_client_.get_num_overall_threads(ec));
    }

    std::uint32_t runtime_distributed::get_num_localities(
        hpx::launch::sync_policy, error_code& ec) const
    {
        return agas_client_.get_num_localities(ec);
    }

    std::uint32_t runtime_distributed::get_initial_num_localities() const
    {
        return get_config().get_num_localities();
    }

    lcos::future<std::uint32_t> runtime_distributed::get_num_localities() const
    {
        return agas_client_.get_num_localities_async();
    }

    std::uint32_t runtime_distributed::get_num_localities(
        hpx::launch::sync_policy, components::component_type type,
        error_code& ec) const
    {
        return agas_client_.get_num_localities(type, ec);
    }

    lcos::future<std::uint32_t> runtime_distributed::get_num_localities(
        components::component_type type) const
    {
        return agas_client_.get_num_localities_async(type);
    }

    std::uint32_t runtime_distributed::assign_cores(
        std::string const& locality_basename, std::uint32_t cores_needed)
    {
        std::lock_guard<std::mutex> l(mtx_);

        used_cores_map_type::iterator it =
            used_cores_map_.find(locality_basename);
        if (it == used_cores_map_.end())
        {
            used_cores_map_.insert(used_cores_map_type::value_type(
                locality_basename, cores_needed));
            return 0;
        }

        std::uint32_t current = (*it).second;
        (*it).second += cores_needed;
        return current;
    }

    std::uint32_t runtime_distributed::assign_cores()
    {
        // adjust thread assignments to allow for more than one locality per
        // node
        std::size_t first_core =
            static_cast<std::size_t>(this->get_config().get_first_used_core());
        std::size_t cores_needed =
            hpx::resource::get_partitioner().assign_cores(first_core);

        return static_cast<std::uint32_t>(cores_needed);
    }

    ///////////////////////////////////////////////////////////////////////////
    void runtime_distributed::default_errorsink(std::string const& msg)
    {
        // log the exception information in any case
        LERR_(always) << "default_errorsink: unhandled exception: " << msg;

        std::cerr << msg << std::endl;
    }

    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_NETWORKING)
    ///////////////////////////////////////////////////////////////////////////
    // Create an instance of a message handler plugin
    void register_message_handler(
        char const* message_handler_type, char const* action, error_code& ec)
    {
        runtime_distributed* rtd = get_runtime_distributed_ptr();
        if (nullptr != rtd)
        {
            return rtd->register_message_handler(
                message_handler_type, action, ec);
        }

        // store the request for later
        get_message_handler_registrations().push_back(
            hpx::util::make_tuple(message_handler_type, action));
    }

    parcelset::policies::message_handler* create_message_handler(
        char const* message_handler_type, char const* action,
        parcelset::parcelport* pp, std::size_t num_messages,
        std::size_t interval, error_code& ec)
    {
        runtime_distributed* rtd = get_runtime_distributed_ptr();
        if (nullptr != rtd)
        {
            return rtd->create_message_handler(
                message_handler_type, action, pp, num_messages, interval, ec);
        }

        HPX_THROWS_IF(ec, invalid_status, "create_message_handler",
            "the runtime system is not available at this time");
        return nullptr;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Create an instance of a binary filter plugin
    serialization::binary_filter* create_binary_filter(
        char const* binary_filter_type, bool compress,
        serialization::binary_filter* next_filter, error_code& ec)
    {
        runtime_distributed* rtd = get_runtime_distributed_ptr();
        if (nullptr != rtd)
            return rtd->create_binary_filter(
                binary_filter_type, compress, next_filter, ec);

        HPX_THROWS_IF(ec, invalid_status, "create_binary_filter",
            "the runtime system is not available at this time");
        return nullptr;
    }
#endif

    runtime_distributed& get_runtime_distributed()
    {
        HPX_ASSERT(get_runtime_distributed_ptr() != nullptr);
        return *get_runtime_distributed_ptr();
    }

    runtime_distributed*& get_runtime_distributed_ptr()
    {
        static thread_local runtime_distributed* runtime_distributed_;
        return runtime_distributed_;
    }

    naming::gid_type const& get_locality()
    {
        return get_runtime_distributed().get_agas_client().get_local_locality();
    }

    ///////////////////////////////////////////////////////////////////////////
    // Helpers
    naming::id_type find_here(error_code& ec)
    {
        if (nullptr == hpx::applier::get_applier_ptr())
        {
            HPX_THROWS_IF(ec, invalid_status, "hpx::find_here",
                "the runtime system is not available at this time");
            return naming::invalid_id;
        }

        static naming::id_type here(
            hpx::applier::get_applier().get_raw_locality(ec),
            naming::id_type::unmanaged);
        return here;
    }

    naming::id_type find_root_locality(error_code& ec)
    {
        runtime_distributed* rt = hpx::get_runtime_distributed_ptr();
        if (nullptr == rt)
        {
            HPX_THROWS_IF(ec, invalid_status, "hpx::find_root_locality",
                "the runtime system is not available at this time");
            return naming::invalid_id;
        }

        naming::gid_type console_locality;
        if (!rt->get_agas_client().get_console_locality(console_locality))
        {
            HPX_THROWS_IF(ec, invalid_status, "hpx::find_root_locality",
                "the root locality is not available at this time");
            return naming::invalid_id;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return naming::id_type(console_locality, naming::id_type::unmanaged);
    }

    std::vector<naming::id_type> find_all_localities(
        components::component_type type, error_code& ec)
    {
        std::vector<naming::id_type> locality_ids;
        if (nullptr == hpx::applier::get_applier_ptr())
        {
            HPX_THROWS_IF(ec, invalid_status, "hpx::find_all_localities",
                "the runtime system is not available at this time");
            return locality_ids;
        }

        hpx::applier::get_applier().get_localities(locality_ids, type, ec);
        return locality_ids;
    }

    std::vector<naming::id_type> find_all_localities(error_code& ec)
    {
        std::vector<naming::id_type> locality_ids;
        if (nullptr == hpx::applier::get_applier_ptr())
        {
            HPX_THROWS_IF(ec, invalid_status, "hpx::find_all_localities",
                "the runtime system is not available at this time");
            return locality_ids;
        }

        hpx::applier::get_applier().get_localities(locality_ids, ec);
        return locality_ids;
    }

    std::vector<naming::id_type> find_remote_localities(
        components::component_type type, error_code& ec)
    {
        std::vector<naming::id_type> locality_ids;
        if (nullptr == hpx::applier::get_applier_ptr())
        {
            HPX_THROWS_IF(ec, invalid_status, "hpx::find_remote_localities",
                "the runtime system is not available at this time");
            return locality_ids;
        }

        hpx::applier::get_applier().get_remote_localities(
            locality_ids, type, ec);
        return locality_ids;
    }

    std::vector<naming::id_type> find_remote_localities(error_code& ec)
    {
        std::vector<naming::id_type> locality_ids;
        if (nullptr == hpx::applier::get_applier_ptr())
        {
            HPX_THROWS_IF(ec, invalid_status, "hpx::find_remote_localities",
                "the runtime system is not available at this time");
            return locality_ids;
        }

        hpx::applier::get_applier().get_remote_localities(
            locality_ids, components::component_invalid, ec);

        return locality_ids;
    }

    // find a locality supporting the given component
    naming::id_type find_locality(
        components::component_type type, error_code& ec)
    {
        if (nullptr == hpx::applier::get_applier_ptr())
        {
            HPX_THROWS_IF(ec, invalid_status, "hpx::find_locality",
                "the runtime system is not available at this time");
            return naming::invalid_id;
        }

        std::vector<naming::id_type> locality_ids;
        hpx::applier::get_applier().get_localities(locality_ids, type, ec);

        if (ec || locality_ids.empty())
            return naming::invalid_id;

        // chose first locality to host the object
        return locality_ids.front();
    }

    std::uint32_t get_num_localities(hpx::launch::sync_policy,
        components::component_type type, error_code& ec)
    {
        runtime_distributed* rt = get_runtime_distributed_ptr();
        if (nullptr == rt)
        {
            HPX_THROW_EXCEPTION(invalid_status, "hpx::get_num_localities",
                "the runtime system has not been initialized yet");
            return 0;
        }

        return rt->get_num_localities(hpx::launch::sync, type, ec);
    }

    lcos::future<std::uint32_t> get_num_localities(
        components::component_type type)
    {
        runtime_distributed* rt = get_runtime_distributed_ptr();
        if (nullptr == rt)
        {
            HPX_THROW_EXCEPTION(invalid_status, "hpx::get_num_localities",
                "the runtime system has not been initialized yet");
            return make_ready_future(std::uint32_t(0));
        }

        return rt->get_num_localities(type);
    }
}    // namespace hpx

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming {
    // shortcut for get_runtime().get_agas_client()
    resolver_client& get_agas_client()
    {
        return get_runtime_distributed().get_agas_client();
    }
}}    // namespace hpx::naming

///////////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_NETWORKING)
namespace hpx { namespace parcelset {
    bool do_background_work(
        std::size_t num_thread, parcelport_background_mode mode)
    {
        return get_runtime_distributed()
            .get_parcel_handler()
            .do_background_work(num_thread, mode);
    }
}}    // namespace hpx::parcelset

#endif
