//  Copyright (c) 2007-2009 Hartmut Kaiser
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
            ("run_agas_server,r", "run AGAS server as part of this runtime instance")
            ("worker,w", "run this instance in worker (non-console) mode")
            ("agas", po::value<std::string>(), 
                "the IP address the AGAS server is running on (default taken "
                "from hpx.ini), expected format: 192.168.1.1:7912")
            ("hpx", po::value<std::string>(), 
                "the IP address the HPX parcelport is listening on (default "
                "is localhost:7910), expected format: 192.168.1.1:7913")
            ("threads,t", po::value<int>(), 
                "the number of operating system threads to spawn for this"
                "HPX locality")
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
class agas_server_helper
{
public:
    agas_server_helper(std::string host, boost::uint16_t port)
      : dgas_pool_(), dgas_(dgas_pool_, host, port)
    {}
    ~agas_server_helper()
    {
        dgas_.stop();
    }

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
    ~test_environment()
    {}

    lcos::counting_semaphore sem_;
    boost::detail::atomic_count counter_;
};

////////////////////////////////////////////////////////////////////////////////
void sem_wait1(boost::shared_ptr<test_environment> env)
{
    ++env->counter_;
    env->sem_.wait();

    // all of the 3 threads need to have incremented the counter
    BOOST_TEST(3 == env->counter_);
}

void sem_signal1(boost::shared_ptr<test_environment> env)
{
    env->sem_.signal(3);    // we need to signal all 3 threads
}

///////////////////////////////////////////////////////////////////////////////
void sem_wait2(boost::shared_ptr<test_environment> env)
{
    // we wait for three other threads to signal this semaphore
    env->sem_.wait(3);

    // all of the 3 threads need to have incremented the counter
    BOOST_TEST(3 == env->counter_);
}

void sem_signal2(boost::shared_ptr<test_environment> env)
{
    ++env->counter_;
    env->sem_.signal();    // we need to signal the semaphore here
}

///////////////////////////////////////////////////////////////////////////////
char const* const sem_wait1_desc[] =
{
    "sem_wait1_1", "sem_wait1_2", "sem_wait1_3"
};

char const* const sem_signal2_desc[] =
{
    "sem_signal2_1", "sem_signal2_2", "sem_signal2_3"
};

int hpx_main()
{
    // create a semaphore, which which we will use to make 3 threads waiting 
    // for a fourth one
    boost::shared_ptr<test_environment> env1(new test_environment);

    // create the  threads which will have to wait on the semaphore
    for (std::size_t i = 0; i < 3; ++i) 
        applier::register_work(boost::bind(&sem_wait1, env1), sem_wait1_desc[i]);

    // now create a thread signaling the semaphore
    applier::register_work(boost::bind(&sem_signal1, env1), "sem_signal1");

    // the 2nd test does the opposite, it creates a semaphore, which which 
    // will be used to make 1 thread waiting for three other threads
    boost::shared_ptr<test_environment> env2(new test_environment);

    // now create a thread waiting on the semaphore
    applier::register_work(boost::bind(&sem_wait2, env2), "sem_wait2");

    // create the threads which will have to signal the semaphore
    for (std::size_t i = 0; i < 3; ++i) 
        applier::register_work(boost::bind(&sem_signal2, env2), sem_signal2_desc[i]);

    // initiate shutdown of the runtime system
    components::stubs::runtime_support::shutdown_all();

    return 0;
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
        hpx::runtime::mode mode = hpx::runtime::console;    // default is console mode
        int num_threads = 0;

        // extract IP address/port arguments
        if (vm.count("agas")) 
            split_ip_address(vm["agas"].as<std::string>(), dgas_host, dgas_port);

        if (vm.count("hpx")) 
            split_ip_address(vm["hpx"].as<std::string>(), hpx_host, hpx_port);

        if (vm.count("worker"))
            mode = hpx::runtime::worker;

        if (vm.count("threads"))
            num_threads = vm["threads"].as<int>();

        // initialize and run the DGAS service, if appropriate
        std::auto_ptr<agas_server_helper> dgas_server;
        if (vm.count("run_agas_server"))  // run the AGAS server instance here
            dgas_server.reset(new agas_server_helper(dgas_host, dgas_port));

        // start the HPX runtime using different numbers of threads
        if (0 == num_threads) {
            hpx::runtime rt(hpx_host, hpx_port, dgas_host, dgas_port, mode);
            for (int i = 1; i <= 8; ++i) 
                rt.run(hpx_main, i);
        }
        else {
            hpx::runtime rt(hpx_host, hpx_port, dgas_host, dgas_port, mode);
            rt.run(hpx_main, num_threads);
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
