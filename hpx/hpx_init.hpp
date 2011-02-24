//  Copyright (c) 2010-2011 Phillip LeBlanc, Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(RUNTIME_HPX_INIT_JUL_1_2010_1118AM)
#define RUNTIME_HPX_INIT_JUL_1_2010_1118AM

#include <iostream>

#include <hpx/hpx.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>

///////////////////////////////////////////////////////////////////////////////
// this function has to be implemented by the user
int hpx_main(boost::program_options::variables_map &vm);

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
  /////////////////////////////////////////////////////////////////////////////
  // Helpers
  inline naming::id_type find_here(void)
  {
    return get_runtime().get_process().here();
  }

  template <typename T>
  inline void 
  get_option(boost::program_options::variables_map& vm, 
    std::string const name, T& x, std::string const& app_name = "")
  {
    if (vm.count(name)) 
      x = vm[name].as<T>();

    if (!app_name.empty()) {
      x = boost::lexical_cast<T>(
          get_runtime().get_config().get_entry(app_name, x));
    }
  }

  template <typename T>
  inline void get_option(T& x, std::string const& app_name)
  {
    if (!app_name.empty()) {
      x = boost::lexical_cast<T>(
          get_runtime().get_config().get_entry(app_name, x));
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  inline void finalize(void)
  {
    double wait_time = 0.0;
    get_option(wait_time, "hpx.finalize_wait_time");

    if (wait_time)
    {
      hpx::util::high_resolution_timer t;
      double start_time = t.elapsed();
      double current = 0;
      do {
        current = t.elapsed();
      } while (current - start_time < wait_time * 1e-6);
    }

    components::stubs::runtime_support::shutdown_all();
  }

  namespace detail
  {
    ///////////////////////////////////////////////////////////////////////////
    // parse the command line
    bool parse_commandline(
        boost::program_options::options_description& app_options, 
        int argc, char *argv[], boost::program_options::variables_map& vm)
    {
      try {
        namespace po = boost::program_options;

        po::options_description hpx_options("HPX Options");
        hpx_options.add_options()
          ("help,h", "print out program usage (this message)")
          ("run_agas_server,r", "run AGAS server as part of this runtime instance")
          ("worker,w", "run this instance in worker (non-console) mode")
          ("config", po::value<std::string>(), 
            "load the specified file as a configuration file")
          ("agas,a", po::value<std::string>(), 
            "the IP address the AGAS server is running on (default taken "
            "from hpx.ini), expected format: 192.168.1.1:7912")
          ("hpx,x", po::value<std::string>(), 
            "the IP address the HPX parcelport is listening on (default "
            "is localhost:7910), expected format: 192.168.1.1:7913")
          ("localities,l", po::value<int>(), 
            "the number of localities to wait for at application startup "
            "(default is 1)")
          ("threads,t", po::value<int>(), 
            "the number of operating system threads to spawn for this "
            "HPX locality")
          ("queueing,q", po::value<std::string>(),
            "the queue scheduling policy to use, options are 'global' "
            " and 'local' (default is 'global')")
        ;

        po::options_description desc_cmdline;
        desc_cmdline.add(app_options).add(hpx_options);

        po::store(
          po::command_line_parser(argc, argv).options(desc_cmdline).run(), vm);
        po::notify(vm);

        // print help screen
        if (vm.count("help")) {
          std::cout << desc_cmdline;
          return false;
        }
      }
      catch (std::exception const& e) {
        std::cerr << "hpx::init: exception caught: " << e.what() << std::endl;
        return false;
      }
      return true;
    }

    ///////////////////////////////////////////////////////////////////////////
    inline void 
    split_ip_address(std::string const& v, std::string& addr, 
      boost::uint16_t& port)
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
        std::cerr << "hpx::init: illegal port number given: " << v.substr(p+1) 
                  << std::endl;
        std::cerr << "           using default value instead: " << port 
                  << std::endl;
      }
    }

    ///////////////////////////////////////////////////////////////////////////
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
  }

  /////////////////////////////////////////////////////////////////////////////
  int init(boost::program_options::options_description& desc_cmdline, 
    int argc, char* argv[])
  {
    int result = 0;
    try
    {
      namespace po = boost::program_options;

      // analyze the command line
      po::variables_map vm;
      if (!detail::parse_commandline(desc_cmdline, argc, argv, vm))
        return -1;

      // Check command line arguments.
      std::string hpx_host("localhost"), agas_host;
      boost::uint16_t hpx_port = HPX_PORT, agas_port = 0;
      int num_threads = 1;
      std::string queueing = "global";
      int num_localities = 1;
      hpx::runtime::mode mode = hpx::runtime::console; // default is console mode

      // extract IP address/port arguments
      if (vm.count("agas")) 
        detail::split_ip_address(vm["agas"].as<std::string>(), agas_host, agas_port);

      if (vm.count("hpx")) 
        detail::split_ip_address(vm["hpx"].as<std::string>(), hpx_host, hpx_port);

      if (vm.count("localities"))
        num_localities = vm["localities"].as<int>();

      if (vm.count("threads"))
        num_threads = vm["threads"].as<int>();

      if (vm.count("queueing"))
        queueing = vm["queueing"].as<std::string>();

      if (vm.count("worker")) {
        mode = hpx::runtime::worker;
        if (vm.count("config")) {
          std::cerr << "hpx::init: --config option ignored, used for console "
            "instance only\n";
        }
      }

      // initialize and run the AGAS service, if appropriate
      std::auto_ptr<detail::agas_server_helper> agas_server;

      // run the AGAS server instance here
      if (vm.count("run_agas_server") || num_localities == 1)  
        agas_server.reset(new detail::agas_server_helper(agas_host, agas_port));

      // initialize and start the HPX runtime
      if (queueing == "global") {
        typedef hpx::threads::policies::global_queue_scheduler global_queue_policy;
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
          result = rt.run(boost::bind(hpx_main, vm), num_threads, num_localities);
        }
        else {
          result = rt.run(num_threads, num_localities);
        }
      }
      else if (queueing == "local") {
        typedef hpx::threads::policies::local_queue_scheduler local_queue_policy;
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
          result = rt.run(boost::bind(hpx_main, vm), num_threads, num_localities);
        }
        else {
          result = rt.run(num_threads, num_localities);
        }
      }
      else {
        throw std::logic_error("bad value for parameter --queuing/-q");
      }
    }
    catch (std::exception& e) {
      std::cerr << "hpx::init: std::exception caught: " << e.what() << "\n";
      return -1;
    }
    catch (...) {
      std::cerr << "hpx::init: unexpected exception caught\n";
      return -2;
    }
    return result;
  }

  /////////////////////////////////////////////////////////////////////////////
  // Minimal initialization function
  int init(std::string const& app_name, int argc = 0, char* argv[] = 0)
  {
    namespace po = boost::program_options;

    std::string desc("usage: ");
    desc += app_name;
    desc += " [options]";
    po::options_description desc_commandline (desc.c_str());
    if (argc == 0 || argv == 0) {
      char *dummy_argv[1] = { const_cast<char*>(app_name.c_str()) };
      return init(desc_commandline, 1, dummy_argv);
    }
    return init(desc_commandline, argc, argv);
  }
}

#endif
