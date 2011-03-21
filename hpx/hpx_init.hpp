//  Copyright (c) 2010-2011 Phillip LeBlanc, Dylan Stark
//  Copyright (c)      2011 Bryce Lelbach
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(ABC9B037_3A25_4591_BB60_CD166773D61D)
#define ABC9B037_3A25_4591_BB60_CD166773D61D

#include <iostream>

#include <boost/shared_ptr.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>

#include <hpx/hpx.hpp>
#include <hpx/util/asio_util.hpp>

///////////////////////////////////////////////////////////////////////////////
// this function has to be implemented by the user
int hpx_main(boost::program_options::variables_map &vm);

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    // Helpers
    inline naming::id_type find_here()
    { return get_runtime().get_process().here(); }

    template <typename T>
    inline void 
    get_option(boost::program_options::variables_map& vm,
               std::string const& name, T& x,
               std::string const& config = "")
    {
        if (vm.count(name)) 
            x = vm[name].as<T>();

        if (!config.empty()) 
            x = boost::lexical_cast<T>
                (get_runtime().get_config().get_entry(config, x));
    }

    template <typename T>
    inline void
    get_option(T& x, std::string const& config)
    {
        if (!config.empty())
            x = boost::lexical_cast<T>
                (get_runtime().get_config().get_entry(config, x));
    }

    ///////////////////////////////////////////////////////////////////////////
    inline void finalize(double wait_time = 0.0, bool user_override = true)
    {
        if (user_override)
          get_option(wait_time, "hpx.finalize_wait_time");

        if (wait_time)
        {
            hpx::util::high_resolution_timer t;
            double start_time = t.elapsed();
            double current = 0.0;
            do {
                current = t.elapsed();
            } while (current - start_time < wait_time * 1e-6);
        }

        components::stubs::runtime_support::shutdown_all();
    }

    namespace detail
    {
        enum command_line_result
        {
            help,
            success,
            error
        }; 

        ///////////////////////////////////////////////////////////////////////
        // parse the command line
        command_line_result parse_commandline(
            boost::program_options::options_description& app_options, 
            int argc, char *argv[], boost::program_options::variables_map& vm)
        {
            using boost::program_options::options_description;
            using boost::program_options::value;
            using boost::program_options::store;
            using boost::program_options::command_line_parser;

            try {
                options_description hpx_options("HPX Options");

                hpx_options.add_options()
                    ("help,h",
                     "print out program usage (this message)")
                    ("run-agas-server,r",
                     "run AGAS server as part of this runtime instance")
                    ("worker,w",
                     "run this instance in worker mode")
                    ("config", value<std::string>(), 
                     "load the specified file as a configuration file")
                    ("agas,a", value<std::string>(), 
                     "the IP address the AGAS server is running on, "
                     "expected format: `address:port' (default: "
                     "taken from hpx.ini)")
                    ("hpx,x", value<std::string>(), 
                     "the IP address the HPX parcelport is listening on, "
                     "expected format: `address:port' (default: "
                     "localhost:7910)")
                    ("random-ports",
                     "use random ports for AGAS and parcels")
                    ("localities,l", value<std::size_t>(), 
                     "the number of localities to wait for at application "
                     "startup (default: 1)")
                    ("threads,t", value<std::size_t>(), 
                     "the number of operating system threads to spawn for this "
                     "HPX locality (default: 1)")
                    ("queueing,q", value<std::string>(),
                     "the queue scheduling policy to use, options are `global' "
                     "and `local' (default: local)")
                ;

                options_description desc_cmdline;
                desc_cmdline.add(app_options).add(hpx_options);

                store(command_line_parser
                    (argc, argv).options(desc_cmdline).run(), vm);
                notify(vm);

                // print help screen
                if (vm.count("help")) {
                    std::cout << desc_cmdline;
                    return help;
                }
            }

            catch (std::exception const& e) {
                std::cerr << "hpx::init: exception caught: "
                          << e.what() << std::endl;
                return error;
            }
            
            return success;
        }

        ///////////////////////////////////////////////////////////////////////
        inline void 
        split_ip_address(std::string const& v, std::string& addr, 
                         boost::uint16_t& port)
        {
            std::string::size_type p = v.find_first_of(":");
            
            try
            {
                if (p != std::string::npos)
                {
                    addr = v.substr(0, p);
                    port = boost::lexical_cast<boost::uint16_t>(v.substr(p+1));
                }
                else
                    addr = v;
            }
            
            catch (boost::bad_lexical_cast const& /*e*/)
            {
                std::cerr << "hpx::init: illegal port number given: "
                          << v.substr(p+1) 
                          << "           using default value instead: "
                          << port 
                          << std::endl;
            }
        }

        ///////////////////////////////////////////////////////////////////////
        // helper class for AGAS server initialization
        class agas_server_helper
        {
          public:
            agas_server_helper(std::string host, boost::uint16_t port)
              : agas_pool_(), agas_(agas_pool_, host, port)
            { agas_.run(false); }

            ~agas_server_helper()
            { agas_.stop(); }
    
          private:
            hpx::util::io_service_pool agas_pool_; 
            hpx::naming::resolver_server agas_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    int init(boost::program_options::options_description& desc_cmdline, 
             int argc, char* argv[])
    {
        int result = 0;
        
        try
        {
            using boost::program_options::variables_map; 
    
            // Analyze the command line.
            variables_map vm;

            switch (detail::parse_commandline(desc_cmdline, argc, argv, vm))
            {
                case detail::error:
                    return 1;
                case detail::help:
                    return 0;
                default:
                    break;
            }  
  
            // Check command line arguments.
            std::string hpx_host("localhost"), agas_host;
            boost::uint16_t hpx_port = HPX_PORT, agas_port = 0;
            std::size_t num_threads = 1;
            std::size_t num_localities = 1;
            std::string queueing = "local";
            hpx::runtime::mode mode = hpx::runtime::console;
    
            if (vm.count("random-ports")
                && !vm.count("agas") && !vm.count("hpx"))
            {
                using boost::fusion::at_c;

                boost::fusion::vector2<boost::uint16_t, boost::uint16_t>
                    ports = hpx::util::get_random_ports();

                std::cout <<   "Randomized port for AGAS: " << at_c<0>(ports)
                          << "\nRandomized port for parcels: " << at_c<1>(ports)
                          << "\n"; 

                agas_port = at_c<0>(ports);
                hpx_port = at_c<1>(ports);
            }

            if (vm.count("agas")) 
                detail::split_ip_address
                    (vm["agas"].as<std::string>(), agas_host, agas_port);
    
            if (vm.count("hpx")) 
                detail::split_ip_address
                    (vm["hpx"].as<std::string>(), hpx_host, hpx_port);
    
            if (vm.count("localities"))
                num_localities = vm["localities"].as<std::size_t>();
    
            if (vm.count("threads"))
                num_threads = vm["threads"].as<std::size_t>();
    
            if (vm.count("queueing"))
                queueing = vm["queueing"].as<std::string>();
    
            if (vm.count("worker"))
            {
                mode = hpx::runtime::worker;
            }
    
            // Initialize and run the AGAS service, if appropriate.
            boost::shared_ptr<detail::agas_server_helper> agas_server;
    
            if (vm.count("run_agas_server") || num_localities == 1)  
                agas_server.reset
                    (new detail::agas_server_helper(agas_host, agas_port));
    
            // Initialize and start the HPX runtime.
            if (queueing == "global")
            {
                typedef hpx::threads::policies::global_queue_scheduler
                    global_queue_policy;
                typedef hpx::runtime_impl<global_queue_policy>
                    runtime_type;
    
                // Build and configure this runtime instance.
                runtime_type rt(hpx_host, hpx_port, agas_host, agas_port, mode);
                if (vm.count("config"))
                {
                    std::string config(vm["config"].as<std::string>());
                    rt.get_config().load_application_configuration
                        (config.c_str());
                }
    
                // Run this runtime instance.
                if (mode != hpx::runtime::worker)
                    result = rt.run(boost::bind
                        (hpx_main, vm), num_threads, num_localities);
                else 
                    result = rt.run(num_threads, num_localities);
            }

            else if (queueing == "local")
            {
                typedef hpx::threads::policies::local_queue_scheduler
                    local_queue_policy;
                typedef hpx::runtime_impl<local_queue_policy> 
                    runtime_type;
    
                local_queue_policy::init_parameter_type init(num_threads, 1000);
    
                // Build and configure this runtime instance.
                runtime_type rt
                    (hpx_host, hpx_port, agas_host, agas_port, mode, init);
                if (vm.count("config"))
                {
                    std::string config(vm["config"].as<std::string>());
                    rt.get_config().load_application_configuration
                        (config.c_str());
                }
    
                // Run this runtime instance.
                if (mode != hpx::runtime::worker)
                    result = rt.run(boost::bind
                        (hpx_main, vm), num_threads, num_localities);
                else
                    result = rt.run(num_threads, num_localities);
            }

            else 
                throw std::logic_error("bad value for parameter --queuing/-q");
        }
        
        catch (std::exception& e)
        {
            std::cerr << "hpx::init: std::exception caught: " << e.what()
                      << "\n";
            return -1;
        }

        catch (...)
        {
            std::cerr << "hpx::init: unexpected exception caught\n";
            return -1;
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////
    int init(int argc = 0, char* argv[] = 0)
    {
        using boost::program_options::options_description; 

        options_description desc_commandline
            ("usage: " HPX_APPLICATION_STRING " [options]");

        if (argc == 0 || argv == 0)
        {
            char *dummy_argv[1] = { const_cast<char*>(HPX_APPLICATION_STRING) };
            return init(desc_commandline, 1, dummy_argv);
        }
    
        return init(desc_commandline, argc, argv);
    }
}

#endif // HPX_ABC9B037_3A25_4591_BB60_CD166773D61D

