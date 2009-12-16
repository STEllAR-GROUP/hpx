//  Copyright (c) 2009-2010 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>

#include <hpx/hpx.hpp>
#include <hpx/runtime/actions/plain_action.hpp>

#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>

#include <hpx/performance_counters/stubs/performance_counter.hpp>

using namespace hpx;
namespace po = boost::program_options;

///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
int monitor(int delay);

typedef 
    actions::plain_result_action1<int, int, monitor>
monitor_action;

HPX_REGISTER_ACTION(monitor_action);

///////////////////////////////////////////////////////////////////////////////

int monitor(int delay)
{
  typedef std::vector<naming::id_type> gids_type;

  hpx::naming::resolver_client const& agas =
      hpx::applier::get_applier().get_agas_client();


  gids_type prefixes;
  applier::get_applier().get_remote_prefixes(prefixes);
  prefixes.push_back(applier::get_applier().get_runtime_support_gid());
  int n_prefixes = prefixes.size();
  
  std::vector<hpx::performance_counters::counter_value> values(n_prefixes);

  for (int i = 0; i < n_prefixes; ++i)
  {
      // Build full performance counter name
      std::string queue("/queue(");
      queue += boost::lexical_cast<std::string>(prefixes[i]);
      queue += "/threadmanager)/length";
    
      // Get GID of perforamnce counter
      hpx::naming::id_type gid;
      agas.queryid(queue, gid);
    
      // Access value of peformance counter
      values[i] = 
          hpx::performance_counters::stubs::
            performance_counter::get_value(gid);
  }

  while (n_prefixes > 0)
  {
    for (int i=0; i<n_prefixes; ++i)
    {
      if (hpx::performance_counters::status_valid_data == values[i].status_
          && values[i].value_ > 0)
        std::cout << "Locale " << prefixes[i] << ": " 
                  << values[i].value_ << std::endl;
    }

    // Busy wait 
    util::high_resolution_timer t;
    double start_time = t.elapsed();
    double current = 0;
    do {
        current = t.elapsed();
    } while (current - start_time < delay*1e-6);
  }


  return 0;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(int delay)
{
    // try to get arguments from application configuration
    runtime& rt = get_runtime();

    // get list of all known localities
    std::vector<naming::id_type> prefixes;
    applier::applier& appl = applier::get_applier();

    naming::id_type this_prefix = appl.get_runtime_support_gid();

    std::cout << "Heart beat monitor, yo!" << std::endl;

    lcos::eager_future<monitor_action> n(this_prefix, delay);
    n.get();
      
    // initiate shutdown of the runtime systems on all localities
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
            ("config", po::value<std::string>(), 
                "load the specified file as an application configuration file")
            ("agas,a", po::value<std::string>(), 
                "the IP address the AGAS server is running on (default taken "
                "from hpx.ini), expected format: 192.168.1.1:7912")
            ("hpx,x", po::value<std::string>(), 
                "the IP address the HPX parcelport is listening on (default "
                "is localhost:7910), expected format: 192.168.1.1:7913")
            ("threads,t", po::value<int>(), 
                "the number of operating system threads to spawn for this"
                "HPX locality")
            ("delay,d", po::value<int>(),
                "the delay at which to ping the system")
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
        std::cerr << "heart_beat: exception caught: " << e.what() << std::endl;
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
        std::cerr << "heart_beat: illegal port number given: " << v.substr(p+1) << std::endl;
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
typedef hpx::runtime_impl<hpx::threads::policies::global_queue_scheduler> runtime_type;

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    try {
        // analyze the command line
        po::variables_map vm;
        if (!parse_commandline(argc, argv, vm))
            return -1;

        // Check command line arguments.
        int delay = 1e6;
        std::string hpx_host("localhost"), agas_host;
        boost::uint16_t hpx_port = HPX_PORT, agas_port = 0;
        int num_threads = 1;
        hpx::runtime::mode mode = hpx::runtime::console;    // default is console mode

        // extract IP address/port arguments
        if (vm.count("agas")) 
            split_ip_address(vm["agas"].as<std::string>(), agas_host, agas_port);

        if (vm.count("hpx")) 
            split_ip_address(vm["hpx"].as<std::string>(), hpx_host, hpx_port);

        if (vm.count("threads"))
            num_threads = vm["threads"].as<int>();

        if (vm.count("worker")) {
            mode = hpx::runtime::worker;
            if (vm.count("config")) {
                std::cerr << "heart_beat: --config option ignored, used for console "
                             "instance only\n";
            }
        }

        if (vm.count("delay")) {
            delay = vm["delay"].as<int>();
        }

        // initialize and run the AGAS service, if appropriate
        std::auto_ptr<agas_server_helper> agas_server;
        if (vm.count("run_agas_server"))  // run the AGAS server instance here
            agas_server.reset(new agas_server_helper(agas_host, agas_port));

        int result = 0;
        double elapsed =0;

        // initialize and start the HPX runtime
        runtime_type rt(hpx_host, hpx_port, agas_host, agas_port, mode);
        if (mode == hpx::runtime::worker) {
            rt.run(num_threads);
        }
        else {
            // if we've got a configuration file (as console) we read it in,
            // otherwise this information will be automatically pulled from 
            // the console
            if (vm.count("config")) {
                std::string config(vm["config"].as<std::string>());
                rt.get_config().load_application_configuration(config.c_str());
            }

            rt.run(boost::bind(hpx_main, delay), num_threads);

        }
    }
    catch (std::exception& e) {
        std::cerr << "heart_beat: std::exception caught: " << e.what() << "\n";
        return -1;
    }
    catch (...) {
        std::cerr << "heart_beat: unexpected exception caught\n";
        return -2;
    }
    return 0;
}

