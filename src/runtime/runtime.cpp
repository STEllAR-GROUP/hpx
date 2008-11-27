//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <vector>

#include <hpx/hpx_fwd.hpp>

#include <boost/config.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>

#include <hpx/include/runtime.hpp>

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
            std::string const& agas_address, boost::uint16_t agas_port) 
      : ini_(), agas_pool_(), parcel_pool_(), timer_pool_(),
        agas_client_(agas_pool_, ini_.get_agas_locality(agas_address, agas_port)),
        parcel_port_(parcel_pool_, naming::locality(address, port)),
        thread_manager_(timer_pool_, 
            boost::bind(&runtime::init_applier, This()),
            boost::bind(&runtime::stop, This(), false)),
        parcel_handler_(agas_client_, parcel_port_, &thread_manager_),
        runtime_support_(ini_, parcel_handler_.get_prefix(), agas_client_),
        applier_(parcel_handler_, thread_manager_, 
            boost::uint64_t(&runtime_support_), boost::uint64_t(&memory_)),
        action_manager_(applier_)
    {}

    ///////////////////////////////////////////////////////////////////////////
    runtime::runtime(naming::locality address, naming::locality agas_address) 
      : ini_(), agas_pool_(), parcel_pool_(), timer_pool_(),
        agas_client_(agas_pool_, agas_address),
        parcel_port_(parcel_pool_, address),
        thread_manager_(timer_pool_, 
            boost::bind(&runtime::init_applier, This()),
            boost::bind(&runtime::stop, This(), false)),
        parcel_handler_(agas_client_, parcel_port_, &thread_manager_),
        runtime_support_(ini_, parcel_handler_.get_prefix(), agas_client_),
        applier_(parcel_handler_, thread_manager_, 
            boost::uint64_t(&runtime_support_), boost::uint64_t(&memory_)),
        action_manager_(applier_)
    {}

    ///////////////////////////////////////////////////////////////////////////
    runtime::runtime(naming::locality address) 
      : ini_(), agas_pool_(), parcel_pool_(), timer_pool_(),
        agas_client_(agas_pool_, ini_.get_agas_locality()),
        parcel_port_(parcel_pool_, address),
        thread_manager_(timer_pool_, 
            boost::bind(&runtime::init_applier, This()),
            boost::bind(&runtime::stop, This(), false)),
        parcel_handler_(agas_client_, parcel_port_, &thread_manager_),
        runtime_support_(ini_, parcel_handler_.get_prefix(), agas_client_),
        applier_(parcel_handler_, thread_manager_, 
            boost::uint64_t(&runtime_support_), boost::uint64_t(&memory_)),
        action_manager_(applier_)
    {}

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
    }

    ///////////////////////////////////////////////////////////////////////////
    void runtime::init_applier()
    {
        applier_.init_tss();
    }

#if defined(_WIN64) && defined(_DEBUG) && !defined(BOOST_COROUTINES_USE_FIBERS)
#include <io.h>
#endif

    ///////////////////////////////////////////////////////////////////////////
    void runtime::start(boost::function<threads::thread_function_type> func, 
        std::size_t num_threads, bool blocking)
    {
#if defined(_WIN64) && defined(_DEBUG) && !defined(BOOST_COROUTINES_USE_FIBERS)
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
        if (!func.empty())
            thread_manager_.register_work(func, "hpx_main");

        LRT_(info) << "runtime: started using "  << num_threads << " OS threads";

        // block if required
        if (blocking) 
            wait();     // wait for the shutdown_action to be executed
    }

    ///////////////////////////////////////////////////////////////////////////
#if !defined(BOOST_WINDOWS)
    static void wait_helper(components::server::runtime_support& rts,
        pthread_t id)
    {
        rts.wait();
        pthread_kill(id, SIGTERM);
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
        // Block all signals for background thread.
        sigset_t new_mask;
        sigfillset(&new_mask);
        sigset_t old_mask;
        pthread_sigmask(SIG_BLOCK, &new_mask, &old_mask);
        pthread_t id = pthread_self();

        // start the wait_helper in a separate thread
        boost::thread t (
            boost::bind(&wait_helper, boost::ref(runtime_support_), id));

        // Restore previous signals.
        pthread_sigmask(SIG_SETMASK, &old_mask, 0);

        // Wait for signal from waiting thread indicating time to shut down.
        sigset_t wait_mask;
        sigemptyset(&wait_mask);
        sigaddset(&wait_mask, SIGINT);
        sigaddset(&wait_mask, SIGQUIT);
        sigaddset(&wait_mask, SIGTERM);
        pthread_sigmask(SIG_BLOCK, &wait_mask, 0);

        // block main thread, this will exit as soon as Ctrl-C has been issued 
        // or any other signal has been received
        int sig = 0;
        sigwait(&wait_mask, &sig);

        t.join();
#endif
        LRT_(info) << "runtime: exiting wait state";
    }

    ///////////////////////////////////////////////////////////////////////////
    void runtime::stop(bool blocking)
    {
        LRT_(info) << "runtime: about to stop services";

        try {
            // unregister the runtime_support and memory instances from the AGAS 
            agas_client_.unbind(applier_.get_runtime_support_gid());
            agas_client_.unbind(applier_.get_memory_gid());
        }
        catch(hpx::exception const&) {
            ; // ignore errors during system shutdown (AGAS might be down already)
        }

        // stop runtime services (threads)
        thread_manager_.stop(blocking);
        parcel_port_.stop(blocking);    // stops parcel_pool_ as well
        agas_pool_.stop();
        runtime_support_.stop();        // re-activate main thread 

        LRT_(info) << "runtime: stopped all services";
    }

    ///////////////////////////////////////////////////////////////////////////
    void runtime::run(boost::function<threads::thread_function_type> func,
        std::size_t num_threads)
    {
        start(func, num_threads);
        wait();
        stop();
    }

    ///////////////////////////////////////////////////////////////////////////
    void runtime::run(std::size_t num_threads)
    {
        start(boost::function<threads::thread_function_type>(), num_threads);
        wait();
        stop();
    }

}

