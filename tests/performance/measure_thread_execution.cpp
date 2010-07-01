//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>

using namespace hpx;
namespace po = boost::program_options;
double globald = 0;

int number_of_iterations = 0;

///////////////////////////////////////////////////////////////////////////////
// this is a empty test thread
threads::thread_state_enum null_thread(threads::thread_state_ex_enum)
{
//     naming::id_type gid = 
//         appl.get_thread_manager().get_thread_gid(self.get_thread_id(), appl);
//     util::high_resolution_timer timer;
    double d = 0.;
    for (int i = 0; i < number_of_iterations; ++i)
    {
        d += 1/(2.* i + 1);
    }
    globald = d;
//     std::cout << timer.elapsed() << std::endl;
    return threads::terminated;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(util::high_resolution_timer& timer, std::size_t num_threads)
{
    // schedule a couple of threads
//     timer.restart();
    for (std::size_t i = 0; i < num_threads; ++i) {
        applier::register_work_plain(null_thread, "null_thread", 
            (naming::address::address_type)null_thread, threads::pending);
    }
//     double elapsed = timer.elapsed();
//     std::cerr << "Elapsed time [s] for thread initialization of " 
//               << num_threads << " threads: " << elapsed << " (" 
//               << elapsed/num_threads << " per thread)" << std::endl;

    // start measuring
    timer.restart();
    applier::get_applier().get_thread_manager().do_some_work();

    // initiate shutdown of the runtime system
    components::stubs::runtime_support::shutdown_all();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
bool parse_commandline(int argc, char *argv[], po::variables_map& vm)
{
    try {
        po::options_description desc_cmdline ("Usage: fibonacci [options]");
        desc_cmdline.add_options()
            ("help,h", "print out program usage (this message)")
            ("run_agas_server,r", "run AGAS server as part of this runtime instance")
            ("worker,w", "run this instance in worker (non-console) mode")
            ("agas,a", po::value<std::string>(), 
                "the IP address the AGAS server is running on (default taken "
                "from hpx.ini), expected format: 192.168.1.1:7912")
            ("hpx,x", po::value<std::string>(), 
                "the IP address the HPX parcelport is listening on (default "
                "is localhost:7910), expected format: 192.168.1.1:7913")
            ("threads,t", po::value<int>(), 
                "the number of operating system threads to spawn for this"
                "HPX locality")
            ("value,v", po::value<int>(), 
                "the number of px-threads to create")
            ("local,l", po::value<int>(), 
                "use local thread scheduler with this number of queues"
                " (default is to use global thread scheduler)")
            ("workload,W", po::value<int>(), 
                "the number of additional iterations creating workload (default: 0)")
            ("numa_sensitive,n", "distribute os-threads across NUMA nodes")
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
        std::cerr << "measure_thread_execution: exception caught: " 
                  << e.what() << std::endl;
        return false;
    }
    return true;
}

///////////////////////////////////////////////////////////////////////////////
inline void 
split_ip_address(std::string const& v, std::string& addr, boost::uint16_t& port)
{
    std::string::size_type p = v.find_first_of(":");
    try {
        if (p != std::string::npos) {
            addr = v.substr(0, p);
            port = boost::lexical_cast<boost::uint16_t>(v.substr(p+1));
        }
        else {
            addr = v;
        }
    }
    catch (boost::bad_lexical_cast const& /*e*/) {
        std::cerr << "measure_thread_execution: illegal port number given: " 
                  << v.substr(p+1) << std::endl;
        std::cerr << "                          using default value instead: " 
                  << port << std::endl;
    }
}

///////////////////////////////////////////////////////////////////////////////
// helper class for AGAS server initialization
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
typedef hpx::runtime_impl<hpx::threads::policies::global_queue_scheduler> 
    global_runtime_type;
typedef hpx::runtime_impl<hpx::threads::policies::local_queue_scheduler> 
    local_runtime_type;

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    try {
        // analyze the command line
        po::variables_map vm;
        if (!parse_commandline(argc, argv, vm))
            return -1;

        // Check command line arguments.
        std::string hpx_host("localhost"), agas_host;
        boost::uint16_t hpx_port = HPX_PORT, agas_port = 0;
        int num_threads = 1;
        int num_hpx_threads = 1;
        hpx::runtime::mode mode = hpx::runtime::console;    // default is console mode

        // extract IP address/port arguments
        if (vm.count("agas")) 
            split_ip_address(vm["agas"].as<std::string>(), agas_host, agas_port);

        if (vm.count("hpx")) 
            split_ip_address(vm["hpx"].as<std::string>(), hpx_host, hpx_port);

        if (vm.count("threads"))
            num_threads = vm["threads"].as<int>();

        if (vm.count("value"))
            num_hpx_threads = vm["value"].as<int>();

        if (vm.count("worker"))
            mode = hpx::runtime::worker;

        if (vm.count("workload"))
            number_of_iterations = vm["workload"].as<int>();

        // initialize and run the AGAS service, if appropriate
        std::auto_ptr<agas_server_helper> agas_server;
        if (vm.count("run_agas_server"))  // run the AGAS server instance here
            agas_server.reset(new agas_server_helper(agas_host, agas_port));

        util::high_resolution_timer timer;
        double elapsed = 0;
        if (!vm.count("local")) {
            // initialize and start the HPX runtime
            global_runtime_type rt(hpx_host, hpx_port, agas_host, agas_port, mode);

            // the main thread will wait (block) for the shutdown action and 
            // the threadmanager is serving incoming requests in the meantime
            rt.run(boost::bind(hpx_main, boost::ref(timer), num_hpx_threads), 
                num_threads);
            elapsed = timer.elapsed();
        }
        else {
            // initialize and start the HPX runtime
            local_runtime_type::scheduling_policy_type::init_parameter_type init(
                /*vm["local"].as<int>()*/num_threads, 0, 
                vm.count("numa_sensitive") != 0);
            local_runtime_type rt(hpx_host, hpx_port, agas_host, agas_port, 
                mode, init);

            // the main thread will wait (block) for the shutdown action and 
            // the threadmanager is serving incoming requests in the meantime
            rt.run(boost::bind(hpx_main, boost::ref(timer), num_hpx_threads), 
                num_threads);
            elapsed = timer.elapsed();
        }
        std::cout << elapsed /*/num_hpx_threads*/ << std::endl;
        std::cout << globald << std::endl;
    }

    catch (std::exception& e) {
        std::cerr << "std::exception caught: " << e.what() << "\n";
        return -1;
    }
    catch (...) {
        std::cerr << "unexpected exception caught\n";
        return -2;
    }
    return 0;
}
