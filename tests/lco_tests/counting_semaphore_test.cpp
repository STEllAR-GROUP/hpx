//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/lcos/counting_semaphore.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/detail/lightweight_test.hpp>
#include <boost/detail/atomic_count.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>

using namespace hpx;
namespace po = boost::program_options;

///////////////////////////////////////////////////////////////////////////////
bool parse_commandline(char const* name, int argc, char *argv[], 
    po::variables_map& vm)
{
    try {
        std::string usage("Usage: ");
        usage += name;
        usage += " [options]";

        po::options_description desc_cmdline (usage);
        desc_cmdline.add_options()
            ("help,h", "print out program usage (this message)")
            ("run_dgas_server,r", "run DGAS server as part of this runtime instance")
            ("dgas", po::value<std::string>(), 
                "the IP address the DGAS server is running on (default taken "
                "from hpx.ini), expected format: 192.168.1.1:7912")
            ("hpx", po::value<std::string>(), 
                "the IP address the HPX parcelport is listening on (default "
                "is localhost:7910), expected format: 192.168.1.1:7913")
        ;

        po::store(po::command_line_parser(argc, argv)
            .options(desc_cmdline).run(), vm);
        po::notify(vm);

        // print help screen
        if (vm.count("help")) {
            std::cout << desc_cmdline;
            return false;
        }
    }
    catch (std::exception const& e) {
        std::cerr << "hpx_runtime: exception caught: " << e.what() << std::endl;
        return false;
    }
    return true;
}

///////////////////////////////////////////////////////////////////////////////
// helper class for DGAS server initialization
class dgas_server_helper
{
public:
    dgas_server_helper(std::string host, boost::uint16_t port)
      : dgas_pool_(), dgas_(dgas_pool_, host, port)
    {}

    void run (bool blocking)
    {
        dgas_.run(blocking);
    }

private:
    hpx::util::io_service_pool dgas_pool_; 
    hpx::naming::resolver_server dgas_;
};

///////////////////////////////////////////////////////////////////////////////
inline void 
split_ip_address(std::string const& v, std::string& addr, boost::uint16_t& port)
{
    try {
        std::string::size_type p = v.find_first_of(":");
        if (p != std::string::npos) {
            addr = v.substr(0, p);
            port = boost::lexical_cast<boost::uint16_t>(v.substr(p+1));
        }
        else {
            addr = v;
        }
    }
    catch (boost::bad_lexical_cast const& /*e*/) {
        ;   // ignore bad_cast exceptions
    }
}

////////////////////////////////////////////////////////////////////////////////
struct test_environment
{
    test_environment()
      : sem_(0), counter_(0)
    {}

    lcos::counting_semaphore sem_;
    boost::detail::atomic_count counter_;
};

////////////////////////////////////////////////////////////////////////////////
threads::thread_state sem_wait1(threads::thread_self& self, 
    applier::applier& appl, boost::shared_ptr<test_environment> env)
{
    ++env->counter_;
    env->sem_.wait(self);

    // all of the 3 threads need to have incremented the counter
    BOOST_TEST(3 == env->counter_);
    return threads::terminated;
}

threads::thread_state sem_signal1(threads::thread_self& self, 
    applier::applier& appl, boost::shared_ptr<test_environment> env)
{
    env->sem_.signal(self, 3);    // we need to signal all 3 threads
    return threads::terminated;
}

///////////////////////////////////////////////////////////////////////////////
threads::thread_state sem_wait2(threads::thread_self& self, 
    applier::applier& appl, boost::shared_ptr<test_environment> env)
{
    // we wait for three other threads to signal this semaphore
    env->sem_.wait(self, 3);

    // all of the 3 threads need to have incremented the counter
    BOOST_TEST(3 == env->counter_);
    return threads::terminated;
}

threads::thread_state sem_signal2(threads::thread_self& self, 
    applier::applier& appl, boost::shared_ptr<test_environment> env)
{
    ++env->counter_;
    env->sem_.signal(self);    // we need to signal the semaphore here
    return threads::terminated;
}

///////////////////////////////////////////////////////////////////////////////
threads::thread_state hpx_main(threads::thread_self& self, 
    applier::applier& appl)
{
    // create a semaphore, which which we will use to make 3 threads waiting 
    // for a fourth one
    boost::shared_ptr<test_environment> env1(new test_environment);
    
    // create the  threads which will have to wait on the semaphore
    threads::threadmanager& tm = appl.get_thread_manager();
    for (std::size_t i = 0; i < 3; ++i) 
        register_work(appl, boost::bind(&sem_wait1, _1, boost::ref(appl), env1));

    // now create a thread signaling the semaphore
    register_work(appl, boost::bind(&sem_signal1, _1, boost::ref(appl), env1));
    
    // the 2nd test does the opposite, it creates a semaphore, which which 
    // will be used to make 1 thread waiting for three other threads
    boost::shared_ptr<test_environment> env2(new test_environment);

    // now create a thread waiting on the semaphore
    register_work(appl, boost::bind(&sem_wait2, _1, boost::ref(appl), env2));

    // create the threads which will have to signal the semaphore
    for (std::size_t i = 0; i < 3; ++i) 
        register_work(appl, boost::bind(&sem_signal2, _1, boost::ref(appl), env2));

    // initiate shutdown of the runtime system
    components::stubs::runtime_support::shutdown_all(appl);

    return threads::terminated;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    try {
        // analyze the command line
        po::variables_map vm;
        if (!parse_commandline("hpx_runtime", argc, argv, vm))
            return -1;

        // Check command line arguments.
        std::string hpx_host("localhost"), dgas_host;
        boost::uint16_t hpx_port = HPX_PORT, dgas_port = 0;

        // extract IP address/port arguments
        if (vm.count("dgas")) 
            split_ip_address(vm["dgas"].as<std::string>(), dgas_host, dgas_port);

        if (vm.count("hpx")) 
            split_ip_address(vm["hpx"].as<std::string>(), hpx_host, hpx_port);

        // initialize and run the DGAS service, if appropriate
        std::auto_ptr<dgas_server_helper> dgas_server;
        if (vm.count("run_dgas_server")) { 
            // run the DGAS server instance here
            dgas_server.reset(new dgas_server_helper(dgas_host, dgas_port));
        }

        // start the HPX runtime using different numbers of threads
        for (int i = 1; i <= 8; ++i) {
            hpx::runtime rt(hpx_host, hpx_port, dgas_host, dgas_port);
            rt.run(hpx_main, i);
        }
    }
    catch (std::exception& e) {
        BOOST_TEST(false);
        std::cerr << "std::exception caught: " << e.what() << "\n";
    }
    catch (...) {
        BOOST_TEST(false);
        std::cerr << "unexpected exception caught\n";
    }
    return boost::report_errors();
}
