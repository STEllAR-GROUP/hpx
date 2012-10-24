//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/config.hpp>
#include <hpx/util/batch_environment.hpp>
#include <hpx/util/map_hostnames.hpp>
#include <hpx/util/sed_transform.hpp>
#include <hpx/util/parse_command_line.hpp>
#include <hpx/util/manage_config.hpp>
#include <hpx/util/command_line_handling.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>
#include <boost/foreach.hpp>
#include <boost/assign/std/vector.hpp>
#include <boost/program_options.hpp>

#include <iostream>
#include <vector>
#include <string>

namespace hpx { namespace util
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        int print_version(std::ostream& out)
        {
            out << std::endl << hpx::copyright() << std::endl;
            out << hpx::complete_version() << std::endl;
            return 0;
        }

        ///////////////////////////////////////////////////////////////////////
        inline void encode (std::string &str, char s, char const *r)
        {
            std::string::size_type pos = 0;
            while ((pos = str.find_first_of(s, pos)) != std::string::npos)
            {
                str.replace (pos, 1, r);
                ++pos;
            }
        }

        inline std::string encode_string(std::string str)
        {
            encode(str, '\n', "\\n");
            return str;
        }

        ///////////////////////////////////////////////////////////////////////
        inline std::string enquote(std::string const& arg)
        {
            if (arg.find_first_of(" \t") != std::string::npos)
                return std::string("\"") + arg + "\"";
            return arg;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    int command_line_handling(
        boost::program_options::options_description& desc_cmdline,
        int argc, char* argv[], std::vector<std::string> ini_config,
        hpx::runtime_mode mode, hpx_main_type& f,
        boost::program_options::variables_map& vm,
        util::runtime_configuration& rtcfg, std::size_t& num_threads,
        std::size_t& num_localities, std::string& queuing)
    {
        // Initial analysis of the command line options. This is
        // preliminary as it will not take into account any aliases as
        // defined in any of the runtime configuration files.
        using boost::program_options::variables_map;
        using namespace boost::assign;

        util::manage_config cfgmap(ini_config);
        bool cmd_result = false;

        {
            // Boost V1.47 and before do not properly reset a variables_map
            // when calling vm.clear(). We work around that problems by
            // creating a separate instance just for the preliminary
            // command line handling.
            variables_map prevm;
            cmd_result = util::parse_commandline(rtcfg, desc_cmdline,
                argc, argv, prevm, util::allow_unregistered, mode);
            if (!cmd_result)
                return -1;

            // re-initialize runtime configuration object
            if (prevm.count("hpx:config"))
                rtcfg.reconfigure(prevm["hpx:config"].as<std::string>());
            else
                rtcfg.reconfigure("");

            // Make sure any aliases defined on the command line get used
            // for the option analysis below.
            std::vector<std::string> cfg;
            if (prevm.count("hpx:ini")) {
                cfg = prevm["hpx:ini"].as<std::vector<std::string> >();
                cfgmap.add(cfg);
            }
            std::copy(ini_config.begin(), ini_config.end(), std::back_inserter(cfg));

            rtcfg.reconfigure(cfg);
        }

        // Re-run program option analysis, ini setting (such as aliases)
        // will be considered now.
        boost::program_options::options_description help;
        std::vector<std::string> unregistered_options;
        cmd_result = util::parse_commandline(rtcfg, desc_cmdline,
            argc, argv, vm, util::allow_unregistered, mode, &help,
            &unregistered_options);
        if (!cmd_result)
            return -1;

        // store unregistered command line arguments
        std::string unregistered_options_cmd_line;
        if (!unregistered_options.empty()) {
            typedef std::vector<std::string>::const_iterator iterator_type;
            iterator_type  end = unregistered_options.end();
            for (iterator_type  it = unregistered_options.begin(); it != end; ++it)
                unregistered_options_cmd_line += " " + detail::enquote(*it);
            ini_config += "hpx.unknown_cmd_line=" +
                detail::enquote(argv[0]) + unregistered_options_cmd_line;
        }

        // print version/copyright information
        if (vm.count("hpx:version"))
            return detail::print_version(std::cout);

        if (vm.count("hpx:help")) {
            std::string help_option(vm["hpx:help"].as<std::string>());
            if (0 == std::string("minimal").find(help_option))
            {
                std::cout << help << std::endl;
                return 0;
            }
            else {
                hpx::util::osstream strm;
                strm << help << std::endl;
                ini_config += "hpx.cmd_line_help=" +
                    detail::encode_string(strm.str());
                ini_config += "hpx.cmd_line_help_option=" + help_option;
            }
        }

        bool debug_clp = vm.count("hpx:debug-clp") ? true : false;

        // create host name mapping
        util::map_hostnames mapnames(debug_clp);
        if (vm.count("hpx:ifsuffix"))
            mapnames.use_suffix(vm["hpx:ifsuffix"].as<std::string>());
        if (vm.count("hpx:ifprefix"))
            mapnames.use_prefix(vm["hpx:ifprefix"].as<std::string>());

        // The AGAS host name and port number are pre-initialized from
        //the command line
        std::string agas_host;
        boost::uint16_t agas_port = HPX_INITIAL_IP_PORT;
        if (vm.count("hpx:agas")) {
            util::split_ip_address(
                vm["hpx:agas"].as<std::string>(), agas_host, agas_port);
        }

        // Check command line arguments.
        util::batch_environment env(debug_clp);

        if (vm.count("hpx:iftransform")) {
            util::sed_transform iftransform(vm["hpx:iftransform"].as<std::string>());

            // Check for parsing failures
            if (!iftransform) {
                throw std::logic_error(boost::str(boost::format(
                    "Could not parse --hpx:iftransform argument '%1%'") %
                    vm["hpx:iftransform"].as<std::string>()));
            }

            typedef util::map_hostnames::transform_function_type
                transform_function_type;
            mapnames.use_transform(transform_function_type(iftransform));
        }

        bool using_nodelist = false;

        if (vm.count("hpx:nodefile")) {
            if (vm.count("hpx:nodes")) {
                throw std::logic_error("Ambiguous command line options. "
                    "Do not specify more than one of the --hpx:nodefile and "
                    "--hpx:nodes options at the same time.");
            }
            using_nodelist = true;
            ini_config += "hpx.nodefile=" +
                env.init_from_file(vm["hpx:nodefile"].as<std::string>(), agas_host);
        }
        else if (vm.count("hpx:nodes")) {
            using_nodelist = true;
            ini_config += "hpx.nodes=" + env.init_from_nodelist(
                vm["hpx:nodes"].as<std::vector<std::string> >(), agas_host);
        }
        // FIXME: What if I don't want to use the node list with SLURM?
        else if (env.found_batch_environment()) {
            using_nodelist = true;
            ini_config += "hpx.nodes=" + env.init_from_environment(agas_host);
        }

        // let the PBS environment decide about the AGAS host
        agas_host = env.agas_host_name(
            agas_host.empty() ? HPX_INITIAL_IP_ADDRESS : agas_host);

        std::string hpx_host(env.host_name(HPX_INITIAL_IP_ADDRESS));
        boost::uint16_t hpx_port = HPX_INITIAL_IP_PORT;

        // handle number of threads
        std::size_t batch_threads = env.retrieve_number_of_threads();

        {
            std::string threads_str
                = cfgmap.get_value<std::string>("hpx.os_threads", "");

            if ("all" == threads_str)
                cfgmap.config_["hpx.os_threads"] =
                    boost::lexical_cast<std::string>(
                        thread::hardware_concurrency());
        }

        num_threads = cfgmap.get_value<std::size_t>("hpx.os_threads", batch_threads);

        if ((env.run_with_pbs() || env.run_with_slurm()) &&
            using_nodelist && (num_threads > batch_threads))
        {
            std::cerr << "hpx::init: command line warning: "
                    "--hpx:ini=hpx.os_threads used when running with "
                << env.get_batch_name()
                << ", requesting a larger number of threads ("
                << num_threads
                << ") than cores have been assigned by "
                << env.get_batch_name()
                << " ("
                << batch_threads
                << "), the application might not run properly."
                << std::endl;
        }

        if (vm.count("hpx:threads")) {
            std::string threads_str = vm["hpx:threads"].as<std::string>();

            std::size_t threads = 0;

            if ("all" == threads_str)
                threads = thread::hardware_concurrency();
            else
                threads = boost::lexical_cast<std::size_t>(threads_str);

            if ((env.run_with_pbs() || env.run_with_slurm()) &&
                using_nodelist && (threads > batch_threads))
            {
                std::cerr << "hpx::init: command line warning: --hpx:threads "
                        "used when running with "
                    << env.get_batch_name() << ", requesting a larger "
                        "number of threads ("
                    << threads
                    << ") than cores have been assigned by "
                    << env.get_batch_name()
                    << " ("
                    << batch_threads
                    << "), the application might not run properly."
                    << std::endl;
            }
            num_threads = threads;
        }

        // handling number of localities
        std::size_t batch_localities = env.retrieve_number_of_localities();
        num_localities = cfgmap.get_value<std::size_t>(
            "hpx.localities", batch_localities);

        if ((env.run_with_pbs() || env.run_with_slurm()) &&
            using_nodelist && (batch_localities != num_localities))
        {
            std::cerr << "hpx::init: command line warning: "
                    "--hpx:ini=hpx.localities used when running with "
                << env.get_batch_name()
                << ", requesting a different number of localities than have "
                    "been assigned by " << env.get_batch_name()
                << ", the application might not run properly."
                << std::endl;
        }

        if (vm.count("hpx:localities")) {
            std::size_t localities = vm["hpx:localities"].as<std::size_t>();
            if ((env.run_with_pbs() || env.run_with_slurm()) &&
                using_nodelist && (localities != num_localities))
            {
                std::cerr << "hpx::init: command line warning: --hpx:localities "
                        "used when running with " << env.get_batch_name()
                    << ", requesting a different "
                        "number of localities than have been assigned by "
                    << env.get_batch_name()
                    << ", the application might not run properly."
                    << std::endl;
            }
            num_localities = localities;
        }

        bool run_agas_server = vm.count("hpx:run-agas-server") ? true : false;
        std::size_t node = env.retrieve_node_number();

        // we initialize certain settings if --node is specified (or data
        // has been retrieved from the environment)
        if (node != std::size_t(-1) || vm.count("hpx:node")) {
            // command line overwrites the environment
            if (vm.count("hpx:node")) {
                if (vm.count("hpx:agas")) {
                    throw std::logic_error("Command line option --hpx:node "
                        "is not compatible with --hpx:agas");
                }
                node = vm["hpx:node"].as<std::size_t>();
            }
            if (env.agas_node() == node) {
                // console node, by default runs AGAS
                run_agas_server = true;
                mode = hpx::runtime_mode_console;
            }
            else if (mode == hpx::runtime_mode_connect) {
                // when connecting we need to select a unique port
                hpx_port = HPX_CONNECTING_IP_PORT;
            }
            else {
                // each node gets an unique port
                hpx_port = static_cast<boost::uint16_t>(hpx_port + node);
                mode = hpx::runtime_mode_worker;

                // do not execute any explicit hpx_main except if asked
                // otherwise
                if (!vm.count("hpx:run-hpx-main"))
                    f = 0;
            }
            // store node number in configuration
            ini_config += "hpx.locality=" +
                boost::lexical_cast<std::string>(node);
        }
        else if (mode == hpx::runtime_mode_connect) {
            // when connecting we need to select a unique port
            hpx_port = HPX_CONNECTING_IP_PORT;
        }

        if (vm.count("hpx:ini")) {
            std::vector<std::string> cfg =
                vm["hpx:ini"].as<std::vector<std::string> >();
            std::copy(cfg.begin(), cfg.end(), std::back_inserter(ini_config));
        }

        if (vm.count("hpx:hpx"))
            util::split_ip_address(vm["hpx:hpx"].as<std::string>(), hpx_host, hpx_port);

        queuing = "priority_local";
        if (vm.count("hpx:queuing"))
            queuing = vm["hpx:queuing"].as<std::string>();

        // If the user has not specified an explicit runtime mode we
        // retrieve it from the command line.
        if (hpx::runtime_mode_default == mode) {
            // The default mode is console, i.e. all workers need to be
            // started with --worker/-w.
            mode = hpx::runtime_mode_console;
            if (vm.count("hpx:console") + vm.count("hpx:worker") +
                vm.count("hpx:connect") > 1)
            {
                throw std::logic_error("Ambiguous command line options. "
                    "Do not specify more than one of --hpx:console, "
                    "--hpx:worker, or --hpx:connect");
            }

            // In these cases we default to executing with an empty
            // hpx_main, except if specified otherwise.
            if (vm.count("hpx:worker")) {
                mode = hpx::runtime_mode_worker;

                // do not execute any explicit hpx_main except if asked
                // otherwise
                if (!vm.count("hpx:run-hpx-main"))
                    f = 0;
            }
            else if (vm.count("hpx:connect")) {
                mode = hpx::runtime_mode_connect;

                // do not execute any explicit hpx_main except if asked
                // otherwise
                if (!vm.count("hpx:run-hpx-main"))
                    f = 0;
            }
        }

        // map host names to ip addresses, if requested
        hpx_host = mapnames.map(hpx_host, hpx_port);
        agas_host = mapnames.map(agas_host, agas_port);

        // sanity checks
        if (num_localities == 1 && !vm.count("hpx:agas") && !vm.count("hpx:node"))
        {
            // We assume we have to run the AGAS server if the number of
            // localities to run on is not specified (or is '1')
            // and no additional option (--hpx:agas or --hpx:node) has been
            // specified. That simplifies running small standalone
            // applications on one locality.
            run_agas_server = (mode != runtime_mode_connect) ? true : false;
        }

        if (hpx_host == agas_host && hpx_port == agas_port) {
            // we assume that we need to run the agas server if the user
            // asked for the same network addresses for HPX and AGAS
            run_agas_server = (mode != runtime_mode_connect) ? true : false;
        }
        else if (run_agas_server) {
            // otherwise, if the user instructed us to run the AGAS server,
            // we set the AGAS network address to the same value as the HPX
            // network address
            agas_host = hpx_host;
            agas_port = hpx_port;
        }
        else if (env.run_with_pbs() || env.run_with_slurm()) {
            // in PBS mode, if the network addresses are different and we
            // should not run the AGAS server we assume to be in worker mode
            mode = hpx::runtime_mode_worker;

            // do not execute any explicit hpx_main except if asked
            // otherwise
            if (!vm.count("hpx:run-hpx-main"))
                f = 0;
        }

        // write HPX and AGAS network parameters to the proper ini-file entries
        ini_config += "hpx.parcel.address=" + hpx_host;
        ini_config += "hpx.parcel.port=" + boost::lexical_cast<std::string>(hpx_port);
        ini_config += "hpx.agas.address=" + agas_host;
        ini_config += "hpx.agas.port=" + boost::lexical_cast<std::string>(agas_port);

        if (run_agas_server) {
            ini_config += "hpx.agas.service_mode=bootstrap";
            if (vm.count("hpx:run-agas-server-only"))
                ini_config += "hpx.components.load_external=0";
        }
        else if (vm.count("hpx:run-agas-server-only") &&
              !(env.run_with_pbs() || env.run_with_slurm()))
        {
            throw std::logic_error("Command line option --hpx:run-agas-server-only "
                "can be specified only for the node running the AGAS server.");
        }
        if (1 == num_localities && vm.count("hpx:run-agas-server-only")) {
            std::cerr << "hpx::init: command line warning: --hpx:run-agas-server-only "
                "used for single locality execution, application might "
                "not run properly." << std::endl;
        }

        // we can't run the AGAS server while connecting
        if (run_agas_server && mode == runtime_mode_connect) {
            throw std::logic_error("Command line option error: can't run AGAS server"
                "while connecting to a running application.");
        }

        // Set whether the AGAS server is running as a dedicated runtime.
        // This decides whether the AGAS actions are executed with normal
        // priority (if dedicated) or with high priority (non-dedicated)
        if (vm.count("hpx:run-agas-server-only"))
            ini_config += "hpx.agas.dedicated_server=1";

        if (vm.count("hpx:debug-hpx-log")) {
            ini_config += "hpx.logging.console.destination=" +
                vm["hpx:debug-hpx-log"].as<std::string>();
            ini_config += "hpx.logging.destination=" +
                vm["hpx:debug-hpx-log"].as<std::string>();
            ini_config += "hpx.logging.console.level=5";
            ini_config += "hpx.logging.level=5";
        }

        if (vm.count("hpx:debug-agas-log")) {
            ini_config += "hpx.logging.console.agas.destination=" +
                vm["hpx:debug-agas-log"].as<std::string>();
            ini_config += "hpx.logging.agas.destination=" +
                vm["hpx:debug-agas-log"].as<std::string>();
            ini_config += "hpx.logging.console.agas.level=5";
            ini_config += "hpx.logging.agas.level=5";
        }

        // Collect the command line for diagnostic purposes.
        std::string cmd_line;
        for (int i = 0; i < argc; ++i)
        {
            // quote only if it contains whitespace
            std::string arg(argv[i]);
            cmd_line += detail::enquote(arg);

            if ((i + 1) != argc)
                cmd_line += " ";
        }

        // Store the program name and the command line.
        ini_config += "hpx.program_name=" + std::string(argv[0]);
        ini_config += "hpx.cmd_line=" + cmd_line;
        ini_config += "hpx.reconstructed_cmd_line=" + 
            detail::enquote(argv[0]) + " " + 
            util::reconstruct_command_line(vm) + " " + 
            unregistered_options_cmd_line;

        // Set number of OS threads in configuration.
        ini_config += "hpx.os_threads=" +
            boost::lexical_cast<std::string>(num_threads);

        // Set number of localities in configuration (do it everywhere,
        // even if this information is only used by the AGAS server).
        ini_config += "hpx.localities=" +
            boost::lexical_cast<std::string>(num_localities);

        // FIXME: AGAS V2: if a locality is supposed to run the AGAS
        //        service only and requests to use 'priority_local' as the
        //        scheduler, switch to the 'local' scheduler instead.
        ini_config += std::string("hpx.runtime_mode=") +
            get_runtime_mode_name(mode);

        if (debug_clp) {
            std::cerr << "Configuration before runtime start:\n";
            std::cerr << "-----------------------------------\n";
            BOOST_FOREACH(std::string const& s, ini_config) {
                std::cerr << s << std::endl;
            }
            std::cerr << "-----------------------------------\n";
        }

        // add all remaining ini settings to the global configuration
        rtcfg.reconfigure(ini_config);

        // all is good
        return 0;
    }
}}

