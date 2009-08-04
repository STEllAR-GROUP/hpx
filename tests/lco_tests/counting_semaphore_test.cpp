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
                "tested semaphores (default: 3, max: 80)")
            ("num_tests,n", po::value<int>(), 
                "the number of times to repeat the test (default: 1)")
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
    test_environment(char const* desc)
      : desc_(desc), sem_(0), counter1_(0)
    {}
    ~test_environment()
    {
        BOOST_TEST(counter1_ == 0);
    }

    std::string desc_;
    lcos::counting_semaphore sem_;
    boost::lockfree::atomic_int<long> counter1_;
};

////////////////////////////////////////////////////////////////////////////////
void sem_wait1(boost::shared_ptr<test_environment> env, int max_semaphore_value)
{
    env->sem_.wait();
    ++env->counter1_;

    // all of the threads need to have incremented the counter
    BOOST_TEST(env->counter1_ <= 0);
}

void sem_signal1(boost::shared_ptr<test_environment> env, int max_semaphore_value)
{
    while (--max_semaphore_value >= 0) {
        --env->counter1_;
        env->sem_.signal();    // we need to signal all threads
    }
}

///////////////////////////////////////////////////////////////////////////////
void sem_wait2(boost::shared_ptr<test_environment> env, int max_semaphore_value)
{
    // we wait for the other threads to signal this semaphore
    while (--max_semaphore_value >= 0) {
        env->sem_.wait();
        --env->counter1_;
    }

    // all of the threads need to have incremented the counter
    BOOST_TEST(0 == env->counter1_);
}

void sem_signal2(boost::shared_ptr<test_environment> env)
{
    ++env->counter1_;
    env->sem_.signal();    // we need to signal the semaphore here
}

///////////////////////////////////////////////////////////////////////////////
char const* const sem_wait1_desc[] =
{
    "sem_wait1_01", "sem_wait1_02", "sem_wait1_03", "sem_wait1_04", "sem_wait1_05",
    "sem_wait1_06", "sem_wait1_07", "sem_wait1_08", "sem_wait1_09", "sem_wait1_10",
    "sem_wait1_11", "sem_wait1_12", "sem_wait1_13", "sem_wait1_14", "sem_wait1_15",
    "sem_wait1_16", "sem_wait1_17", "sem_wait1_18", "sem_wait1_19", "sem_wait1_20",
    "sem_wait1_21", "sem_wait1_22", "sem_wait1_23", "sem_wait1_24", "sem_wait1_25",
    "sem_wait1_26", "sem_wait1_27", "sem_wait1_28", "sem_wait1_29", "sem_wait1_30",
    "sem_wait1_31", "sem_wait1_32", "sem_wait1_33", "sem_wait1_34", "sem_wait1_35",
    "sem_wait1_36", "sem_wait1_37", "sem_wait1_38", "sem_wait1_39", "sem_wait1_40",
    "sem_wait1_41", "sem_wait1_42", "sem_wait1_43", "sem_wait1_44", "sem_wait1_45",
    "sem_wait1_46", "sem_wait1_47", "sem_wait1_48", "sem_wait1_49", "sem_wait1_50",
    "sem_wait1_51", "sem_wait1_52", "sem_wait1_53", "sem_wait1_54", "sem_wait1_55",
    "sem_wait1_56", "sem_wait1_57", "sem_wait1_58", "sem_wait1_59", "sem_wait1_60",
    "sem_wait1_61", "sem_wait1_62", "sem_wait1_63", "sem_wait1_64", "sem_wait1_65",
    "sem_wait1_66", "sem_wait1_67", "sem_wait1_68", "sem_wait1_69", "sem_wait1_70",
    "sem_wait1_71", "sem_wait1_72", "sem_wait1_73", "sem_wait1_74", "sem_wait1_75",
    "sem_wait1_76", "sem_wait1_77", "sem_wait1_78", "sem_wait1_79", "sem_wait1_80"
};

char const* const sem_signal2_desc[] =
{
    "sem_signal21_01", "sem_signal21_02", "sem_signal21_03", "sem_signal21_04", "sem_signal21_05",
    "sem_signal21_06", "sem_signal21_07", "sem_signal21_08", "sem_signal21_09", "sem_signal21_10",
    "sem_signal21_11", "sem_signal21_12", "sem_signal21_13", "sem_signal21_14", "sem_signal21_15",
    "sem_signal21_16", "sem_signal21_17", "sem_signal21_18", "sem_signal21_19", "sem_signal21_20",
    "sem_signal21_21", "sem_signal21_22", "sem_signal21_23", "sem_signal21_24", "sem_signal21_25",
    "sem_signal21_26", "sem_signal21_27", "sem_signal21_28", "sem_signal21_29", "sem_signal21_30",
    "sem_signal21_31", "sem_signal21_32", "sem_signal21_33", "sem_signal21_34", "sem_signal21_35",
    "sem_signal21_36", "sem_signal21_37", "sem_signal21_38", "sem_signal21_39", "sem_signal21_40",
    "sem_signal21_41", "sem_signal21_42", "sem_signal21_43", "sem_signal21_44", "sem_signal21_45",
    "sem_signal21_46", "sem_signal21_47", "sem_signal21_48", "sem_signal21_49", "sem_signal21_50",
    "sem_signal21_51", "sem_signal21_52", "sem_signal21_53", "sem_signal21_54", "sem_signal21_55",
    "sem_signal21_56", "sem_signal21_57", "sem_signal21_58", "sem_signal21_59", "sem_signal21_60",
    "sem_signal21_61", "sem_signal21_62", "sem_signal21_63", "sem_signal21_64", "sem_signal21_65",
    "sem_signal21_66", "sem_signal21_67", "sem_signal21_68", "sem_signal21_69", "sem_signal21_70",
    "sem_signal21_71", "sem_signal21_72", "sem_signal21_73", "sem_signal21_74", "sem_signal21_75",
    "sem_signal21_76", "sem_signal21_77", "sem_signal21_78", "sem_signal21_79", "sem_signal21_80"
};

int hpx_main(std::size_t max_semaphore_value)
{
    ///////////////////////////////////////////////////////////////////////////
    // create a semaphore, which which we will use to make several threads 
    // waiting for another one
    boost::shared_ptr<test_environment> env1(new test_environment("test1"));

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
    boost::shared_ptr<test_environment> env2(new test_environment("test2"));

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
    boost::shared_ptr<test_environment> env3(new test_environment("test3"));

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
    boost::shared_ptr<test_environment> env4(new test_environment("test4"));

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
        int num_tests = 1;

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

        if (vm.count("num_tests"))
            num_tests = vm["num_tests"].as<int>();

        if (max_semaphore_value > 80)
            max_semaphore_value = 80;

        // initialize and run the DGAS service, if appropriate
        std::auto_ptr<agas_server_helper> dgas_server;
        if (vm.count("run_agas_server"))  // run the AGAS server instance here
            dgas_server.reset(new agas_server_helper(dgas_host, dgas_port));

        // start the HPX runtime using different numbers of threads
        if (0 == num_threads) {
            hpx::runtime rt(hpx_host, hpx_port, dgas_host, dgas_port, mode);
            for (int i = 0; i < num_tests; ++i) {
                for (int i = 1; i <= 8; ++i) { 
                    rt.run(boost::bind(hpx_main, max_semaphore_value), i);
                    std::cerr << ".";
                }
            }
            std::cerr << "\n";
        }
        else {
            hpx::runtime rt(hpx_host, hpx_port, dgas_host, dgas_port, mode);
            for (int i = 0; i < num_tests; ++i) 
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
