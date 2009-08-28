//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/lcos/barrier.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/detail/lightweight_test.hpp>
#include <boost/detail/atomic_count.hpp>
#include <boost/program_options.hpp>

using namespace hpx;
namespace po = boost::program_options;

///////////////////////////////////////////////////////////////////////////////
void barrier_test(lcos::barrier& b, boost::detail::atomic_count& c, std::size_t count)
{
    ++c;
    b.wait();

    // all of the 'count' threads need to have incremented the counter
    BOOST_TEST(count == c);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(lcos::barrier& b, boost::detail::atomic_count& c, std::size_t count)
{
    // create the 4 threads which will have to wait on the barrier
    for (std::size_t i = 0; i < count; ++i) {
        applier::register_work(
            boost::bind(&barrier_test, boost::ref(b), boost::ref(c), count));
    }

    b.wait();     // wait for all threads to enter the barrier

    // all of the 'count' threads need to have incremented the counter
    BOOST_TEST(count == c);

    // initiate shutdown of the runtime system
    components::stubs::runtime_support::shutdown_all();

    return 0;
}

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

///////////////////////////////////////////////////////////////////////////////
// this is the runtime type we use in this application
typedef hpx::runtime_impl<hpx::threads::policies::global_queue_scheduler> runtime_type;

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

        if (vm.count("num_tests"))
            num_tests = vm["num_tests"].as<int>();

        // initialize and run the DGAS service, if appropriate
        std::auto_ptr<agas_server_helper> dgas_server;
        if (vm.count("run_agas_server"))  // run the AGAS server instance here
            dgas_server.reset(new agas_server_helper(dgas_host, dgas_port));

        // start the HPX runtime using different numbers of threads
        if (0 == num_threads) {
            int num_of_cores = boost::thread::hardware_concurrency();
            runtime_type rt(hpx_host, hpx_port, dgas_host, dgas_port, mode);
            for (int t = 0; t < num_tests; ++t) {
                for (int i = 1; i <= 2*num_of_cores; ++i) { 
                    std::size_t count = 2 * i;
                    lcos::barrier b(count+1);       // create a barrier waiting on 'count' threads
                    boost::detail::atomic_count c(0);

                    rt.run(boost::bind(hpx_main, boost::ref(b), boost::ref(c), count), i);
                    std::cerr << ".";
                }
            }
            std::cerr << "\n";
        }
        else {
            runtime_type rt(hpx_host, hpx_port, dgas_host, dgas_port, mode);
            for (int t = 0; t < num_tests; ++t) {
                std::size_t count = 2 * num_threads;
                lcos::barrier b(count + 1);       // create a barrier waiting on 'count' threads
                boost::detail::atomic_count c(0);

                rt.run(boost::bind(hpx_main, boost::ref(b), boost::ref(c), count), num_threads);
                std::cerr << ".";
            }
            std::cerr << "\n";
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
