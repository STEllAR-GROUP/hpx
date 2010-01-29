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

#include "heart_beat.hpp"

using namespace hpx;
namespace po = boost::program_options;

///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// This is one long-running thread.
//
// Currently only monitors "this" locality, and never stops :-).

HPX_REGISTER_ACTION(monitor_action);

int monitor(double frequency, double duration, double rate)
{
  typedef hpx::naming::id_type gid_type;
  typedef std::vector<gid_type> gids_type;
  typedef util::high_resolution_timer timer_type;

  hpx::naming::resolver_client const& agas =
      hpx::applier::get_applier().get_agas_client();

  gid_type here = applier::get_applier().get_runtime_support_gid();

  // Build full performance counter name
  std::string queue("/queue(");
  queue += boost::lexical_cast<std::string>(here);
  queue += "/threadmanager)/length";

  // Get GID of performance counter
  gid_type gid;
  agas.queryid(queue, gid);

  std::cout << "Begin timing block" << std::endl;

  // Start segment
  timer_type t;
  double current_time;

  while(true)
  {
      std::cout << "\tBegin segment" << std::endl;

      double segment_start = t.elapsed();
      do {
          std::cout << "\t\tBegin monitoring block" << std::endl;

          // Start monitoring phase
          double monitor_start = t.elapsed();
          do{
              current_time = t.elapsed();

              // Access value of performance counter
              hpx::performance_counters::counter_value value;
              value =
                  hpx::performance_counters::stubs::
                    performance_counter::get_value(gid);

              if (hpx::performance_counters::status_valid_data == value.status_)
              {
                  std::cout << current_time << ": " << value.value_ << std::endl;
              }

              // Adjust rate of pinging values
              double delay_start = t.elapsed();
              do {
                  current_time = t.elapsed();
              } while(current_time - delay_start < rate);
          } while (current_time - monitor_start < duration);
      } while (current_time - segment_start < frequency);

      // Adjust rate of monitoring phases
      double pause_start = t.elapsed();
      do {
          current_time = t.elapsed();
      } while(current_time - pause_start < (frequency-duration));
  }

  return 0;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(double delay, double frequency, double duration, double rate)
{
    // Delay
    util::high_resolution_timer t;
    double start_time = t.elapsed();
    double current = 0;
    do {
        current = t.elapsed();
    } while (current - start_time < delay);

    naming::id_type here = applier::get_applier().get_runtime_support_gid();

    std::cout << "Heart beat monitor, yo!" << std::endl;

    lcos::eager_future<monitor_action> n(here,
                                         duration,
                                         frequency,
                                         rate);
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
            ("frequency,f", po::value<int>(),
                "how often to start monitoring (ms)")
            ("duration,d", po::value<int>(),
                "how long to monitor (ms)")
            ("rate", po::value<int>(),
                "how often to poll (ms)")
            ("delay,p", po::value<int>(),
                "the amount of time to delay before monitoring starts (s) "
                "(gives time to start other application)")
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
        int frequency = 1000000;
        int duration  =  500000;
        int rate      =    1000;

        int delay = 1;

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

        if (vm.count("frequency")) {
            frequency = vm["frequency"].as<int>();
        }
        if (vm.count("duration")) {
            duration = vm["duration"].as<int>();
        }
        if (vm.count("rate")) {
            rate = vm["rate"].as<int>();
        }

        if (vm.count("delay")) {
            delay = vm["delay"].as<int>();
        }

        // initialize and run the AGAS service, if appropriate
        std::auto_ptr<agas_server_helper> agas_server;
        if (vm.count("run_agas_server"))  // run the AGAS server instance here
            agas_server.reset(new agas_server_helper(agas_host, agas_port));

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

            rt.run(boost::bind(hpx_main, delay*1.0e-6,
                               frequency*1.0e-6,
                               duration*1.0e-6,
                               rate*1.0e-6),
                   num_threads);

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

