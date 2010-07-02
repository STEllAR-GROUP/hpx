//  Copyright (c) 2010-2011 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>

#include <hpx/hpx.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>

#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>

using namespace hpx;
namespace po = boost::program_options;

///////////////////////////////////////////////////////////////////////////////
// Helpers
typedef hpx::naming::id_type id_type;
typedef hpx::naming::gid_type gid_type;

inline gid_type find_here(void)
{
  return hpx::applier::get_applier().get_runtime_support_raw_gid();
}

///////////////////////////////////////////////////////////////////////////////
// More helpers
template<typename Action, typename Arg0>
inline void apply(Arg0 arg0, id_type k)
{
  naming::id_type here(find_here(), naming::id_type::unmanaged);
  hpx::applier::apply<Action>(here, arg0, k);
}

template<typename Action, typename Arg0, typename Arg1>
inline void apply(Arg0 arg0, Arg1 arg1, id_type k)
{
  naming::id_type here(find_here(), naming::id_type::unmanaged);
  hpx::applier::apply<Action>(here, arg0, arg1, k);
}

template<typename Arg0>
inline void trigger(id_type k, Arg0 arg0)
{
  actions::continuation(k).trigger<Arg0>(arg0);
}

///////////////////////////////////////////////////////////////////////////////
// The following CPS example was adapted from 
// http://en.wikipedia.org/wiki/Continuation-passing_style (Jun 30, 2010).
//
// def factorial(n, k):
//     factorial_aux(n, 1, k)
//
// def factorial_aux(n, a, k):
//     if n == 0:
//         k(a)
//     else:
//         factorial_aux(n-1, n*a, k)
//
// >>> k = dataflow_variable()
// >>> factorial(3, k)
//
// >>> print k
// 6

///////////////////////////////////////////////////////////////////////////////
void factorial_aux(int, int, id_type);
typedef actions::plain_action3<int, int, id_type, factorial_aux> 
    factorial_aux_action;
HPX_REGISTER_PLAIN_ACTION(factorial_aux_action);

void factorial(int, id_type);
typedef actions::plain_action2<int, id_type, factorial> 
    factorial_action;
HPX_REGISTER_PLAIN_ACTION(factorial_action);

void factorial_aux(int n, int a, id_type k)
{
  if (n == 0)
    trigger<int>(k, a);
  else
    apply<factorial_aux_action>(n-1, n*a, k);
}

void factorial(int n, id_type k)
{
  apply<factorial_aux_action>(n, 1, k);
}

///////////////////////////////////////////////////////////////////////////////
typedef hpx::lcos::dataflow_variable<int,int> dataflow_int_type;

///////////////////////////////////////////////////////////////////////////////
int hpx_main(po::variables_map &vm)
{
    int n = -1;
    if (vm.count("value"))
      n = vm["value"].as<int>();

    {
        std::cout << ">>> k1 = dataflow_variable()" << std::endl;
        dataflow_int_type k1;

        std::cout << ">>> factorial(0, k1)" << std::endl;
        apply<factorial_action>(0, k1.get_gid());

        std::cout << ">>> print k1" << std::endl;
        std::cout << k1.get() << std::endl;

        std::cout << ">>> k2 = dataflow()" << std::endl;
        dataflow_int_type k2;

        std::cout << ">>> factorial(1, k2)" << std::endl;
        apply<factorial_action>(1, k2.get_gid());

        std::cout << ">>> print k2" << std::endl;
        std::cout << k2.get() << std::endl;

        std::cout << ">>> k3 = dataflow()" << std::endl;
        dataflow_int_type k3;

        std::cout << ">>> factorial(3, k3)" << std::endl;
        apply<factorial_action>(3, k3.get_gid());

        std::cout << ">>> print k3" << std::endl;
        std::cout << k3.get() << std::endl;

        if (n >= 0)
        {
          std::cout << ">>> kn = dataflow()" << std::endl;
          dataflow_int_type kn;

          std::cout << ">>> factorial(" << n << ", kn)" << std::endl;
          apply<factorial_action>(n, kn.get_gid());

          std::cout << ">>> print kn" << std::endl;
          std::cout << kn.get() << std::endl;
        }
    }

    // initiate shutdown of the runtime systems on all localities
    components::stubs::runtime_support::shutdown_all();

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
bool parse_commandline(int argc, char *argv[], po::variables_map& vm)
{
    try {
        po::options_description desc_cmdline ("Usage: fibonacci2 [options]");
        desc_cmdline.add_options()
            ("help,h", "print out program usage (this message)")
            ("run_agas_server,r", "run AGAS server as part of this runtime instance")
            ("worker,w", "run this instance in worker (non-console) mode")
            ("config", po::value<std::string>(), 
                "load the specified file as an application configuration file")
            ("agas,a", po::value<std::string>(), 
                "the IP address the AGAS server is running on (default taken "
                "from hpx.ini), expected format: 192.168.1.1:7912")
            ("hpx,x", po::value<std::string>(), 
                "the IP address the HPX parcelport is listening on (default "
                "is localhost:7910), expected format: 192.168.1.1:7913")
            ("localities,l", po::value<int>(), 
                "the number of localities to wait for at application startup"
                "(default is 1)")
            ("threads,t", po::value<int>(), 
                "the number of operating system threads to spawn for this"
                "HPX locality")
            ("queueing,q", po::value<std::string>(),
                "the queue scheduling policy to use, options are 'global' "
                " and 'local' (default is 'global')")
            ("value,v", po::value<int>(), 
                "the number to be used as the argument to fib (default is 10)")
            ("csv,s", "generate statistics of the run in comma separated format")
            ("busywait,b", po::value<int>(),
                "add this amount of busy wait workload to each of the iterations"
                " [in steps of 1µs], i.e. -b1000 == 1ms")
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
        std::cerr << "fibonacci2: exception caught: " << e.what() << std::endl;
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
        std::cerr << "fibonacci2: illegal port number given: " << v.substr(p+1) << std::endl;
        std::cerr << "           using default value instead: " << port << std::endl;
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
// this is the runtime type we use in this application
typedef hpx::threads::policies::global_queue_scheduler global_queue_policy;
typedef hpx::threads::policies::local_queue_scheduler local_queue_policy;

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
        std::string queueing = "global";
        int num_localities = 1;
        hpx::runtime::mode mode = hpx::runtime::console;    // default is console mode

        // extract IP address/port arguments
        if (vm.count("agas")) 
            split_ip_address(vm["agas"].as<std::string>(), agas_host, agas_port);

        if (vm.count("hpx")) 
            split_ip_address(vm["hpx"].as<std::string>(), hpx_host, hpx_port);

        if (vm.count("localities"))
            num_localities = vm["localities"].as<int>();

        if (vm.count("threads"))
            num_threads = vm["threads"].as<int>();

        if (vm.count("queueing"))
            queueing = vm["queueing"].as<std::string>();

        if (vm.count("worker")) {
            mode = hpx::runtime::worker;
            if (vm.count("config")) {
                std::cerr << "fibonacci2: --config option ignored, used for console "
                             "instance only\n";
            }
        }

        // initialize and run the AGAS service, if appropriate
        std::auto_ptr<agas_server_helper> agas_server;
        if (vm.count("run_agas_server"))  // run the AGAS server instance here
            agas_server.reset(new agas_server_helper(agas_host, agas_port));

        // initialize and start the HPX runtime
        if (queueing == "global")
        {
            typedef hpx::runtime_impl<global_queue_policy> runtime_type;

            // Build and configure this runtime instance
            runtime_type rt(hpx_host, hpx_port, agas_host, agas_port, mode);
            if (mode != hpx::runtime::worker && vm.count("config"))
            {
                std::string config(vm["config"].as<std::string>());
                rt.get_config().load_application_configuration(config.c_str());
            }
               
            // Run this runtime instance
            if (mode != hpx::runtime::worker) {
                rt.run(boost::bind(hpx_main, vm), num_threads, num_localities);
            }
            else
            {
                rt.run(num_threads, num_localities);
            }
        }
        else if (queueing == "local")
        {
            typedef hpx::runtime_impl<local_queue_policy> runtime_type;
            local_queue_policy::init_parameter_type init(num_threads, 1000);

            // Build and configure this runtime instance
            runtime_type rt(hpx_host, hpx_port, agas_host, agas_port, mode, init);
            if (mode != hpx::runtime::worker && vm.count("config"))
            {
                std::string config(vm["config"].as<std::string>());
                rt.get_config().load_application_configuration(config.c_str());
            }
               
            // Run this runtime instance
            if (mode != hpx::runtime::worker) {
                rt.run(boost::bind(hpx_main, vm), num_threads, num_localities);
            }
            else
            {
                rt.run(num_threads, num_localities);
            }
        }
        else
            BOOST_ASSERT(false);
    }
    catch (std::exception& e) {
        std::cerr << "fibonacci2: std::exception caught: " << e.what() << "\n";
        return -1;
    }
    catch (...) {
        std::cerr << "fibonacci2: unexpected exception caught\n";
        return -2;
    }
    return 0;
}

