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
            ("value,v", po::value<int>(), 
                "the number of threads to create for concurrent access to the "
                "tested semaphores (default: 3, max: 20)")
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
      : agas_pool_(), agas_(agas_pool_, host, port)
    {
        agas_.run(false);
    }
    ~agas_server_helper()
    {
        agas_.stop();
    }

private:
    hpx::util::io_service_pool agas_pool_; 
    hpx::naming::resolver_server agas_;
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
    test_environment(char const* desc, int max_semaphore_value)
      : desc_(desc),
        sem_(0), counter_(0), max_semaphore_value_(max_semaphore_value)
    {}
    ~test_environment()
    {
        BOOST_ASSERT(counter_ == max_semaphore_value_);
        BOOST_ASSERT(0 == sem_.get_value());
    }

    std::string desc_;
    lcos::counting_semaphore sem_;
    int max_semaphore_value_;
    boost::detail::atomic_count counter_;
};

////////////////////////////////////////////////////////////////////////////////
void sem_wait1(boost::shared_ptr<test_environment> env, int max_semaphore_value)
{
    ++env->counter_;
    env->sem_.wait();

    // all of the threads need to have incremented the counter, or some of the
    // threads are still sitting in the semaphore
    BOOST_TEST(max_semaphore_value == env->counter_ ||
               max_semaphore_value == env->counter_ + env->sem_.get_value());
}

void sem_signal1(boost::shared_ptr<test_environment> env, int max_semaphore_value)
{
    env->sem_.signal(max_semaphore_value);    // we need to signal all threads
}

///////////////////////////////////////////////////////////////////////////////
void sem_wait2(boost::shared_ptr<test_environment> env, int max_semaphore_value)
{
    // we wait for the other threads to signal this semaphore
    env->sem_.wait(max_semaphore_value);

    // all of the threads need to have incremented the counter
    BOOST_TEST(max_semaphore_value == env->counter_);
}

void sem_signal2(boost::shared_ptr<test_environment> env)
{
    ++env->counter_;
    env->sem_.signal();    // we need to signal the semaphore here
}

///////////////////////////////////////////////////////////////////////////////
char const* const sem_wait1_desc[] =
{
    "sem_wait1_01", "sem_wait1_02", "sem_wait1_03", "sem_wait1_04", "sem_wait1_05",
    "sem_wait1_06", "sem_wait1_07", "sem_wait1_08", "sem_wait1_09", "sem_wait1_10",
    "sem_wait1_11", "sem_wait1_12", "sem_wait1_13", "sem_wait1_14", "sem_wait1_15",
    "sem_wait1_16", "sem_wait1_17", "sem_wait1_18", "sem_wait1_19", "sem_wait1_20"
};

char const* const sem_signal2_desc[] =
{
    "sem_signal21_01", "sem_signal21_02", "sem_signal21_03", "sem_signal21_04", "sem_signal21_05",
    "sem_signal21_06", "sem_signal21_07", "sem_signal21_08", "sem_signal21_09", "sem_signal21_10",
    "sem_signal21_11", "sem_signal21_12", "sem_signal21_13", "sem_signal21_14", "sem_signal21_15",
    "sem_signal21_16", "sem_signal21_17", "sem_signal21_18", "sem_signal21_19", "sem_signal21_20"
};

int hpx_main(std::size_t max_semaphore_value)
{
    ///////////////////////////////////////////////////////////////////////////
    // create a semaphore, which which we will use to make several threads 
    // waiting for another one
    boost::shared_ptr<test_environment> env1(
        new test_environment("test1", max_semaphore_value));

    // create the  threads which will have to wait on the semaphore
    for (std::size_t i = 0; i < max_semaphore_value; ++i) 
    {
        applier::register_work(boost::bind(&sem_wait1, env1, max_semaphore_value), 
            sem_wait1_desc[i]);
    }

    // now create a thread signaling the semaphore
    applier::register_work(boost::bind(&sem_signal1, env1, max_semaphore_value), 
        "sem_signal1");

    ///////////////////////////////////////////////////////////////////////////
    // create a semaphore, which we will use to make several threads 
    // waiting for another one, but the semaphore is signaled before being 
    // waited on
    boost::shared_ptr<test_environment> env2(
        new test_environment("test2", max_semaphore_value));

    // create a thread signaling the semaphore
    applier::register_work(boost::bind(&sem_signal1, env2, max_semaphore_value), 
        "sem_signal1");

    // create the  threads which will have to wait on the semaphore
    for (std::size_t i = 0; i < max_semaphore_value; ++i) 
    {
        applier::register_work(boost::bind(&sem_wait1, env2, max_semaphore_value), 
            sem_wait1_desc[i]);
    }

    ///////////////////////////////////////////////////////////////////////////
    // the 3rd test does the opposite, it creates a semaphore, which  
    // will be used to make one thread waiting for several other threads
    boost::shared_ptr<test_environment> env3(
        new test_environment("test3", max_semaphore_value));

    // now create a thread waiting on the semaphore
    applier::register_work(boost::bind(&sem_wait2, env3, max_semaphore_value), 
        "sem_wait2");

    // create the threads which will have to signal the semaphore
    for (std::size_t i = 0; i < max_semaphore_value; ++i) 
        applier::register_work(boost::bind(&sem_signal2, env3), sem_signal2_desc[i]);

    ///////////////////////////////////////////////////////////////////////////
    // the 4th test does the opposite, it creates a semaphore, which  
    // will be used to make one thread waiting for several other threads, but 
    // the semaphore is signaled before being waited on
    boost::shared_ptr<test_environment> env4(
        new test_environment("test4", max_semaphore_value));

    // create the threads which will have to signal the semaphore
    for (std::size_t i = 0; i < max_semaphore_value; ++i) 
        applier::register_work(boost::bind(&sem_signal2, env4), sem_signal2_desc[i]);

    // now create a thread waiting on the semaphore
    applier::register_work(boost::bind(&sem_wait2, env4, max_semaphore_value), 
        "sem_wait2");

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
        std::size_t max_semaphore_value = 3;

        // extract IP address/port arguments
        if (vm.count("agas")) 
            split_ip_address(vm["agas"].as<std::string>(), dgas_host, dgas_port);

        if (vm.count("hpx")) 
            split_ip_address(vm["hpx"].as<std::string>(), hpx_host, hpx_port);

        if (vm.count("worker"))
            mode = hpx::runtime::worker;

        if (vm.count("threads"))
            num_threads = vm["threads"].as<int>();

        if (vm.count("value"))
            max_semaphore_value = vm["value"].as<int>();

        if (max_semaphore_value > 20)
            max_semaphore_value = 20;

        // initialize and run the DGAS service, if appropriate
        std::auto_ptr<agas_server_helper> dgas_server;
        if (vm.count("run_agas_server"))  // run the AGAS server instance here
            dgas_server.reset(new agas_server_helper(dgas_host, dgas_port));

        // start the HPX runtime using different numbers of threads
        if (0 == num_threads) {
            hpx::runtime rt(hpx_host, hpx_port, dgas_host, dgas_port, mode);
            for (int i = 1; i <= 8; ++i) 
                rt.run(boost::bind(hpx_main, max_semaphore_value), i);
        }
        else {
            hpx::runtime rt(hpx_host, hpx_port, dgas_host, dgas_port, mode);
            rt.run(boost::bind(hpx_main, max_semaphore_value), num_threads);
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
