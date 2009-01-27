//  Copyright (c) 2007-2009 Hartmut Kaiser
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
#include <hpx/runtime/components/server/console_error_sink_singleton.hpp>

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
    runtime::runtime(std::string const& address, boost::uint16_t port,
            std::string const& agas_address, boost::uint16_t agas_port, 
            mode locality_mode, 
            boost::function<hpx_errorsink_function_type> errorsink) 
      : mode_(locality_mode), ini_(util::detail::get_logging_data()), 
        agas_pool_(), parcel_pool_(), timer_pool_(),
        agas_client_(agas_pool_, ini_.get_agas_locality(agas_address, agas_port), mode_ == console),
        parcel_port_(parcel_pool_, naming::locality(address, port)),
        thread_manager_(timer_pool_, 
            boost::bind(&runtime::init_applier, This()),
            boost::bind(&runtime::stop, This(), false),
            boost::bind(&runtime::report_error, This(), _1)),
        parcel_handler_(agas_client_, parcel_port_, &thread_manager_),
        init_logging_(ini_, mode_ == console, agas_client_, parcel_handler_.get_prefix()),
        applier_(parcel_handler_, thread_manager_, 
            boost::uint64_t(&runtime_support_), boost::uint64_t(&memory_)),
        action_manager_(applier_),
        runtime_support_(ini_, parcel_handler_.get_prefix(), agas_client_, applier_)
    {
        if (errorsink) {
            components::server::get_error_dispatcher().register_error_sink(
                errorsink, error_sink_);
        }
        else {
            components::server::get_error_dispatcher().register_error_sink(
                &runtime::default_errorsink, error_sink_);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    runtime::runtime(naming::locality address, naming::locality agas_address, 
            mode locality_mode, 
            boost::function<hpx_errorsink_function_type> errorsink) 
      : mode_(locality_mode), ini_(util::detail::get_logging_data()), 
        agas_pool_(), parcel_pool_(), timer_pool_(),
        agas_client_(agas_pool_, agas_address, mode_ == console),
        parcel_port_(parcel_pool_, address),
        thread_manager_(timer_pool_, 
            boost::bind(&runtime::init_applier, This()),
            boost::bind(&runtime::stop, This(), false),
            boost::bind(&runtime::report_error, This(), _1)),
        parcel_handler_(agas_client_, parcel_port_, &thread_manager_),
        init_logging_(ini_, mode_ == console, agas_client_, parcel_handler_.get_prefix()),
        applier_(parcel_handler_, thread_manager_, 
            boost::uint64_t(&runtime_support_), boost::uint64_t(&memory_)),
        action_manager_(applier_),
        runtime_support_(ini_, parcel_handler_.get_prefix(), agas_client_, applier_)
    {
        if (errorsink) {
            components::server::get_error_dispatcher().register_error_sink(
                errorsink, error_sink_);
        }
        else {
            components::server::get_error_dispatcher().register_error_sink(
                &runtime::default_errorsink, error_sink_);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    runtime::runtime(naming::locality address, mode locality_mode, 
            boost::function<hpx_errorsink_function_type> errorsink) 
      : mode_(locality_mode), ini_(util::detail::get_logging_data()), 
        agas_pool_(), parcel_pool_(), timer_pool_(),
        agas_client_(agas_pool_, ini_.get_agas_locality(), mode_ == console),
        parcel_port_(parcel_pool_, address),
        thread_manager_(timer_pool_, 
            boost::bind(&runtime::init_applier, This()),
            boost::bind(&runtime::stop, This(), false),
            boost::bind(&runtime::report_error, This(), _1)),
        parcel_handler_(agas_client_, parcel_port_, &thread_manager_),
        init_logging_(ini_, mode_ == console, agas_client_, parcel_handler_.get_prefix()),
        applier_(parcel_handler_, thread_manager_, 
            boost::uint64_t(&runtime_support_), boost::uint64_t(&memory_)),
        action_manager_(applier_),
        runtime_support_(ini_, parcel_handler_.get_prefix(), agas_client_, applier_)
    {
        if (errorsink) {
            components::server::get_error_dispatcher().register_error_sink(
                errorsink, error_sink_);
        }
        else {
            components::server::get_error_dispatcher().register_error_sink(
                &runtime::default_errorsink, error_sink_);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    runtime::~runtime()
    {
        LRT_(debug) << "~runtime(entering)";

        // stop all services
        parcel_port_.stop();      // stops parcel_pool_ as well
        thread_manager_.stop();   // stops timer_pool_ as well
        agas_pool_.stop();

        runtime_support_.tidy();  // unload libraries

        LRT_(debug) << "~runtime(finished)";

        // this disables all logging from the main thread
        applier_.deinit_tss();    // reset thread specific pointer
    }

    ///////////////////////////////////////////////////////////////////////////
    void runtime::init_applier()
    {
        applier_.init_tss();
    }

#if defined(_WIN64) && defined(_DEBUG) && !defined(BOOST_COROUTINE_USE_FIBERS)
#include <io.h>
#endif

    ///////////////////////////////////////////////////////////////////////////
    static threads::thread_state 
    run_helper(boost::function<runtime::hpx_main_function_type> func, 
        int& result)
    {
        result = func();
        return threads::terminated;
    }

    int runtime::start(boost::function<hpx_main_function_type> func, 
        std::size_t num_threads, bool blocking)
    {
#if defined(_WIN64) && defined(_DEBUG) && !defined(BOOST_COROUTINE_USE_FIBERS)
        // needs to be called to avoid problems at system startup
        // see: http://connect.microsoft.com/VisualStudio/feedback/ViewFeedback.aspx?FeedbackID=100319
        _isatty(0);
#endif

        // start services (service threads)
        thread_manager_.run(num_threads);   // start the thread manager, timer_pool_ as well
        parcel_port_.run(false);            // starts parcel_pool_ as well

        // register the runtime_support and memory instances with the AGAS 
        agas_client_.bind(applier_.get_runtime_support_gid(), 
            naming::address(parcel_port_.here(), 
                components::get_component_type<components::server::runtime_support>(), 
                &runtime_support_));

        agas_client_.bind(applier_.get_memory_gid(), 
            naming::address(parcel_port_.here(), 
                components::get_component_type<components::server::memory>(), 
                &memory_));

        // register the given main function with the thread manager
        int result = 0;
        if (!func.empty()) {
            thread_manager_.register_thread(
                boost::bind(run_helper, func, boost::ref(result)), "hpx_main");
        }

        LRT_(info) << "runtime: started using "  << num_threads << " OS threads";

        // block if required
        if (blocking) 
            wait();     // wait for the shutdown_action to be executed

        return result;
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

    void runtime::wait()
    {
        LRT_(info) << "runtime: about to enter wait state";

#if defined(BOOST_WINDOWS)
        // Set console control handler to allow server to be stopped.
        console_ctrl_function = boost::bind(&runtime::stop, this, true);
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
        LRT_(info) << "runtime: exiting wait state";
    }

    ///////////////////////////////////////////////////////////////////////////
    void runtime::stop(bool blocking)
    {
        LRT_(info) << "runtime: about to stop services";

        // unregister the runtime_support and memory instances from the AGAS 
        // ignore errors, as AGAS might be down already
        error_code ec;
        agas_client_.unbind(applier_.get_runtime_support_gid(), ec);
        agas_client_.unbind(applier_.get_memory_gid(), ec);

        // stop runtime services (threads)
        thread_manager_.stop(blocking);
        parcel_port_.stop(blocking);    // stops parcel_pool_ as well
        agas_pool_.stop();
        runtime_support_.stop();        // re-activate main thread 

        LRT_(info) << "runtime: stopped all services";
    }

    ///////////////////////////////////////////////////////////////////////////
    void runtime::report_error(boost::exception_ptr const& e)
    {
        // first report this error to the console
        naming::id_type console_prefix;
        if (agas_client_.get_console_prefix(console_prefix))
        {
            components::console_error_sink(console_prefix, 
                parcel_handler_.get_prefix(), e);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    int runtime::run(boost::function<hpx_main_function_type> func,
        std::size_t num_threads)
    {
        int result = start(func, num_threads);
        wait();
        stop();
        return result;
    }

    ///////////////////////////////////////////////////////////////////////////
    int runtime::run(std::size_t num_threads)
    {
        int result = start(boost::function<hpx_main_function_type>(), num_threads);
        wait();
        stop();
        return result;
    }

    ///////////////////////////////////////////////////////////////////////////
    void runtime::default_errorsink(boost::uint32_t src, 
        boost::exception_ptr const& e)
    {
        try {
            boost::rethrow_exception(e);
        }
        catch (boost::exception const& be) {
            std::cerr << "locality(" << std::hex << std::setw(4) 
                      << std::setfill('0') << src << "):" << std::endl
                      << boost::diagnostic_information(be) << std::endl;
        }
    }

}

