//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <vector>

#include <hpx/hpx_fwd.hpp>

#include <boost/config.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/assign/std/vector.hpp>
#include <boost/preprocessor/stringize.hpp>

#include <hpx/include/runtime.hpp>
#include <hpx/util/init_ini_data.hpp>

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
    runtime::runtime(std::string const& address, unsigned short port,
            std::string const& dgas_address, unsigned short dgas_port) 
      : ini_(), dgas_pool_(), parcel_pool_(), timer_pool_(),
        dgas_client_(dgas_pool_, ini_.get_dgas_locality(dgas_address, dgas_port)),
        parcel_port_(parcel_pool_, address, port),
        thread_manager_(timer_pool_),
        parcel_handler_(dgas_client_, parcel_port_, &thread_manager_),
        runtime_support_(ini_, dgas_client_),
        applier_(parcel_handler_, thread_manager_, 
            boost::uint64_t(&runtime_support_), boost::uint64_t(&memory_)),
        action_manager_(applier_)
    {}

    ///////////////////////////////////////////////////////////////////////////
    runtime::runtime(naming::locality address, naming::locality dgas_address) 
      : ini_(), dgas_pool_(), parcel_pool_(), timer_pool_(),
        dgas_client_(dgas_pool_, dgas_address),
        parcel_port_(parcel_pool_, address),
        thread_manager_(timer_pool_),
        parcel_handler_(dgas_client_, parcel_port_, &thread_manager_),
        runtime_support_(ini_, dgas_client_),
        applier_(parcel_handler_, thread_manager_, 
            boost::uint64_t(&runtime_support_), boost::uint64_t(&memory_)),
        action_manager_(applier_)
    {}

    ///////////////////////////////////////////////////////////////////////////
    runtime::runtime(naming::locality address) 
      : ini_(), dgas_pool_(), parcel_pool_(), timer_pool_(),
        dgas_client_(dgas_pool_, ini_.get_dgas_locality()),
        parcel_port_(parcel_pool_, address),
        thread_manager_(timer_pool_),
        parcel_handler_(dgas_client_, parcel_port_, &thread_manager_),
        runtime_support_(ini_, dgas_client_),
        applier_(parcel_handler_, thread_manager_, 
            boost::uint64_t(&runtime_support_), boost::uint64_t(&memory_)),
        action_manager_(applier_)
    {}

    ///////////////////////////////////////////////////////////////////////////
    runtime::~runtime()
    {
        // stop all services
        parcel_port_.stop();      // stops parcel_pool_ as well
        thread_manager_.stop();   // stops timer_pool_ as well
        dgas_pool_.stop();
    }

#if defined(_WIN64) && defined(_DEBUG) && !defined(BOOST_COROUTINES_USE_FIBERS)
#include <io.h>
#endif

    ///////////////////////////////////////////////////////////////////////////
    void runtime::start(boost::function<hpx_main_function_type> func, 
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

        // register the runtime_support and memory instances with the DGAS 
        dgas_client_.bind(applier_.get_runtime_support_gid(), 
            naming::address(parcel_port_.here(), 
                components::server::runtime_support::get_component_type(), 
                &runtime_support_));

        dgas_client_.bind(applier_.get_memory_gid(), 
            naming::address(parcel_port_.here(), 
                components::server::memory::get_component_type(), &memory_));

        // register the given main function with the thread manager
        if (!func.empty())
        {
            thread_manager_.register_work(
                boost::bind(func, _1, boost::ref(applier_)));
        }

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
    }

    ///////////////////////////////////////////////////////////////////////////
    void runtime::stop(bool blocking)
    {
        try {
            // unregister the runtime_support and memory instances from the DGAS 
            naming::id_type factoryid (parcel_handler_.get_prefix().get_msb()+1, 0);
            dgas_client_.unbind(factoryid);
            dgas_client_.unbind(parcel_handler_.get_prefix());
        }
        catch(hpx::exception const&) {
            ; // ignore errors during system shutdown (DGAS might be down already)
        }

        // stop runtime services (threads)
        thread_manager_.stop(blocking);
        parcel_port_.stop(blocking);    // stops parcel_pool_ as well
        dgas_pool_.stop();
        runtime_support_.stop();        // re-activate main thread 
    }

    ///////////////////////////////////////////////////////////////////////////
    void runtime::run(boost::function<hpx_main_function_type> func,
        std::size_t num_threads)
    {
        start(func, num_threads);
        wait();
        stop();
    }

    ///////////////////////////////////////////////////////////////////////////
    void runtime::run(std::size_t num_threads)
    {
        start(boost::function<hpx_main_function_type>(), num_threads);
        wait();
        stop();
    }

    ///////////////////////////////////////////////////////////////////////////
    runtime::runtime_config::runtime_config()
    {
        // pre-initialize entries with compile time based values
        using namespace boost::assign;
        std::vector<std::string> lines; 
        lines +=
                "[hpx]",
                "location = " HPX_PREFIX,
                "ini_path = $[hpx.location]/share/hpx/ini",
                "dgas_address = ${HPX_DGAS_SERVER_ADRESS:" HPX_NAME_RESOLVER_ADDRESS "}",
                "dgas_port = ${HPX_DGAS_SERVER_PORT:" BOOST_PP_STRINGIZE(HPX_NAME_RESOLVER_PORT) "}"
            ;
        this->parse("static defaults", lines);

        // try to build default ini structure from shared libraries in default 
        // installation location, this allows to install simple components
        // without the need to install an ini file
        util::init_ini_data_default(HPX_DEFAULT_COMPONENT_PATH, *this);

        // add explicit configuration information if its provided
        if (util::init_ini_data_base(*this)) {
            // merge all found ini files of all components
            util::merge_component_inis(*this);

            // read system and user ini files _again_, to allow the user to 
            // overwrite the settings from the default component ini's. 
            util::init_ini_data_base(*this);
        }
    }

    // DGAS configuration information has to be stored in the global HPX 
    // configuration section:
    // 
    //    [hpx]
    //    dgas_address=<ip address>   # this defaults to HPX_NAME_RESOLVER_ADDRESS
    //    dgas_port=<ip port>         # this defaults to HPX_NAME_RESOLVER_PORT
    //
    naming::locality runtime::runtime_config::get_dgas_locality()
    {
        // load all components as described in the configuration information
        if (has_section("hpx")) {
            util::section* sec = get_section("hpx");
            if (NULL != sec) {
                std::string cfg_port(
                    sec->get_entry("dgas_port", HPX_NAME_RESOLVER_PORT));

                return naming::locality(
                    sec->get_entry("dgas_address", HPX_NAME_RESOLVER_ADDRESS),
                    boost::lexical_cast<unsigned short>(cfg_port));
            }
        }
        return naming::locality(HPX_NAME_RESOLVER_ADDRESS, HPX_NAME_RESOLVER_PORT);
    }

    naming::locality runtime::runtime_config::get_dgas_locality(
        std::string default_address, unsigned short default_port)
    {
        // load all components as described in the configuration information
        if (has_section("hpx")) {
            util::section* sec = get_section("hpx");
            if (NULL != sec) {
                // read fall back values from cfg file, if needed
                if (default_address.empty()) {
                    default_address = 
                        sec->get_entry("dgas_address", HPX_NAME_RESOLVER_ADDRESS);
                }
                if (-1 == default_port) {
                    default_port = boost::lexical_cast<unsigned short>(
                        sec->get_entry("dgas_port", HPX_NAME_RESOLVER_PORT));
                }
                return naming::locality(default_address, default_port);
            }
        }
        return naming::locality(HPX_NAME_RESOLVER_ADDRESS, HPX_NAME_RESOLVER_PORT);
    }

}

