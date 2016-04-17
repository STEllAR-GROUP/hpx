//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/config.hpp>
#include <hpx/config/asio.hpp>
#include <hpx/version.hpp>
#include <hpx/util/asio_util.hpp>
#include <hpx/util/batch_environment.hpp>
#include <hpx/util/map_hostnames.hpp>
#include <hpx/util/sed_transform.hpp>
#include <hpx/util/parse_command_line.hpp>
#include <hpx/util/manage_config.hpp>
#include <hpx/util/command_line_handling.hpp>
#include <hpx/util/detail/reset_function.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/runtime/threads/policies/affinity_data.hpp>
#include <hpx/runtime/threads/policies/topology.hpp>

#include <boost/asio/ip/host_name.hpp>
#include <boost/assign/std/vector.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <boost/shared_ptr.hpp>

#include <iostream>
#include <string>
#include <vector>

namespace hpx { namespace util
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        int print_version(std::ostream& out)
        {
            out << std::endl << hpx::copyright() << std::endl;
            out << hpx::complete_version() << std::endl;
            return 1;
        }

        int print_info(std::ostream& out, util::command_line_handling const& cfg)
        {
            out << "Static configuration:\n---------------------\n";
            out << hpx::configuration_string() << std::endl;

            out << "Runtime configuration:\n----------------------\n";
            out << hpx::runtime_configuration_string(cfg) << std::endl;

            return 1;
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
        void report_thread_warning(std::string const& batch_name,
            std::size_t threads, std::size_t batch_threads)
        {
            std::cerr << "hpx::init: command line warning: --hpx:threads "
                    "used when running with "
                << batch_name
                << ", requesting a larger number of threads ("
                << threads
                << ") than cores have been assigned by "
                << batch_name
                << " ("
                << batch_threads
                << "), the application might not run properly."
                << std::endl;
        }

        void report_locality_warning_batch(std::string const& batch_name,
            std::size_t batch_localities, std::size_t num_localities)
        {
            std::cerr << "hpx::init: command line warning: "
                    "--hpx:localities used when running with "
                << batch_name
                << ", requesting a different number of localities ("
                << num_localities
                << ") than have been assigned by "
                << batch_name
                << " ("
                << batch_localities
                << "), the application might not run properly."
                << std::endl;
        }

        void report_locality_warning(std::string const& batch_name,
            std::size_t cmdline_localities, std::size_t num_localities)
        {
            std::cerr << "hpx::init: command line warning: "
                    "--hpx:localities used when running with "
                << batch_name
                << ", requesting a different number of localities ("
                << num_localities
                << ") than have been assigned on the command line "
                << " ("
                << cmdline_localities
                << "), the application might not run properly."
                << std::endl;
        }

        std::string convert_to_log_file(std::string const& dest)
        {
            if (dest.empty())
                return "cout";

            if (dest == "cout" || dest == "cerr" || dest == "console")
                return dest;
#if defined(ANDROID) || defined(__ANDROID__)
            if (dest == "android_log")
                return dest;
#endif
            // everything else is assumed to be a file name
            return "file(" + dest + ")";
        }

        ///////////////////////////////////////////////////////////////////////
        std::size_t handle_num_localities(util::manage_config& cfgmap,
            boost::program_options::variables_map& vm,
            util::batch_environment& env, bool using_nodelist,
            std::size_t num_localities)
        {
            std::size_t batch_localities = env.retrieve_number_of_localities();
            if (num_localities == 1 && batch_localities != std::size_t(-1))
            {
                std::size_t cfg_num_localities = cfgmap.get_value<std::size_t>(
                    "hpx.localities", batch_localities);
                if (cfg_num_localities > 1)
                    num_localities = cfg_num_localities;
            }

            if (env.found_batch_environment() &&
                using_nodelist && (batch_localities != num_localities) &&
                (num_localities != 1))
            {
                detail::report_locality_warning_batch(env.get_batch_name(),
                    batch_localities, num_localities);
            }

            if (vm.count("hpx:localities")) {
                std::size_t localities = vm["hpx:localities"].as<std::size_t>();

                if (localities == 0)
                {
                    throw hpx::detail::command_line_error(
                        "Number of --hpx:localities must be greater than 0");
                }

                if (env.found_batch_environment() &&
                    using_nodelist && (localities != num_localities) &&
                    (num_localities != 1))
                {
                    detail::report_locality_warning(env.get_batch_name(),
                        localities, num_localities);
                }
                num_localities = localities;
            }
            return num_localities;
        }

        std::string handle_queueing(util::manage_config& cfgmap,
            boost::program_options::variables_map& vm, std::string default_)
        {
            // command line options is used preferred
            if (vm.count("hpx:queuing"))
                return vm["hpx:queuing"].as<std::string>();

            // use either cfgmap value or default
            return cfgmap.get_value<std::string>("hpx.scheduler", default_);
        }

        std::string handle_affinity(util::manage_config& cfgmap,
            boost::program_options::variables_map& vm, std::string default_)
        {
            // command line options is used preferred
            if (vm.count("hpx:affinity"))
                return vm["hpx:affinity"].as<std::string>();

            // use either cfgmap value or default
            return cfgmap.get_value<std::string>("hpx.affinity", default_);
        }

        std::string handle_affinity_bind(util::manage_config& cfgmap,
            boost::program_options::variables_map& vm, std::string default_)
        {
            // command line options is used preferred
            if (vm.count("hpx:bind"))
            {
                std::string affinity_desc;

                std::vector<std::string> bind_affinity =
                    vm["hpx:bind"].as<std::vector<std::string> >();
                for (std::string const& s : bind_affinity)
                {
                    if (!affinity_desc.empty())
                        affinity_desc += ";";
                    affinity_desc += s;
                }

                return affinity_desc;
            }

            // use either cfgmap value or default
            return cfgmap.get_value<std::string>("hpx.bind", default_);
        }

        std::size_t handle_pu_step(util::manage_config& cfgmap,
            boost::program_options::variables_map& vm, std::size_t default_)
        {
            // command line options is used preferred
            if (vm.count("hpx:pu-step"))
                return vm["hpx:pu-step"].as<std::size_t>();

            // use either cfgmap value or default
            return cfgmap.get_value<std::size_t>("hpx.pu_step", default_);
        }

        std::size_t handle_pu_offset(util::manage_config& cfgmap,
            boost::program_options::variables_map& vm, std::size_t default_)
        {
            // command line options is used preferred
            if (vm.count("hpx:pu-offset"))
                return vm["hpx:pu-offset"].as<std::size_t>();

            // use either cfgmap value or default
            return cfgmap.get_value<std::size_t>("hpx.pu_offset", default_);
        }

        std::size_t handle_numa_sensitive(util::manage_config& cfgmap,
            boost::program_options::variables_map& vm, std::size_t default_)
        {
            if (vm.count("hpx:numa-sensitive") != 0)
            {
                std::size_t numa_sensitive =
                    vm["hpx:numa-sensitive"].as<std::size_t>();
                if (numa_sensitive > 2)
                {
                    throw hpx::detail::command_line_error("Invalid argument "
                        "value for --hpx:numa-sensitive. Allowed values are "
                        "0, 1, or 2");
                }
                return numa_sensitive;
            }

            // use either cfgmap value or default
            return cfgmap.get_value<std::size_t>("hpx.numa_sensitive", default_);
        }

        ///////////////////////////////////////////////////////////////////////
        std::size_t handle_num_threads(util::manage_config& cfgmap,
            boost::program_options::variables_map& vm,
            util::batch_environment& env, bool using_nodelist)
        {
            std::size_t batch_threads = env.retrieve_number_of_threads();
            std::size_t default_threads = 1;
            std::string threads_str = cfgmap.get_value<std::string>(
                "hpx.os_threads", "");

            if ("all" == threads_str)
            {
                if (batch_threads == std::size_t(-1))
                    batch_threads = thread::hardware_concurrency();
                else
                    default_threads = batch_threads;

                cfgmap.config_["hpx.os_threads"] =
                    std::to_string(batch_threads);
            }
            else if (batch_threads != std::size_t(-1))
            {
                default_threads = batch_threads;
            }

            std::size_t threads = cfgmap.get_value<std::size_t>(
                "hpx.os_threads", default_threads);

            if (vm.count("hpx:threads"))
            {
                threads_str = vm["hpx:threads"].as<std::string>();
                if ("all" == threads_str)
                {
                    if (batch_threads == std::size_t(-1))
                    {
                        batch_threads = thread::hardware_concurrency();
                    }
                    threads = batch_threads; //-V101
                }
                else
                {
                    threads = hpx::util::safe_lexical_cast<std::size_t>(threads_str);
                }

                if (threads == 0)
                {
                    throw hpx::detail::command_line_error("Number of --hpx:threads "
                        "must be greater than 0");
                }

#if defined(HPX_HAVE_MAX_CPU_COUNT)
                if (threads > HPX_HAVE_MAX_CPU_COUNT)
                {
                    throw hpx::detail::command_line_error("Requested more than "
                        BOOST_PP_STRINGIZE(HPX_HAVE_MAX_CPU_COUNT)" --hpx:threads "
                        "to use for this application, use the option "
                        "-DHPX_WITH_MAX_CPU_COUNT=<N> when configuring HPX.");
                }
#endif
            }

            if (env.found_batch_environment() &&
                using_nodelist && (threads > batch_threads))
            {
                detail::report_thread_warning(env.get_batch_name(),
                    threads, batch_threads);
            }
            return threads;
        }

        ///////////////////////////////////////////////////////////////////////
        std::size_t get_number_of_default_cores(util::batch_environment& env)
        {
            threads::topology& top = threads::create_topology();

            std::size_t batch_threads = env.retrieve_number_of_threads();
            std::size_t num_cores = top.get_number_of_cores();
            if(batch_threads == std::size_t(-1))
                return num_cores;

            // assuming we assign the first N cores ...
            std::size_t core = 0;
            for(; core < num_cores; ++core)
            {
                batch_threads -= top.get_number_of_core_pus(core);
                if(batch_threads == 0) break;
            }

            return core + 1;
        }

        std::size_t handle_num_cores(util::manage_config& cfgmap,
            boost::program_options::variables_map& vm, std::size_t num_threads,
            util::batch_environment& env)
        {
            std::string cores_str = cfgmap.get_value<std::string>("hpx.cores", "");
            if ("all" == cores_str) {
                cfgmap.config_["hpx.cores"] = std::to_string(
                    get_number_of_default_cores(env));
            }

            std::size_t num_cores = cfgmap.get_value<std::size_t>("hpx.cores",
                num_threads);
            if (vm.count("hpx:cores")) {
                cores_str = vm["hpx:cores"].as<std::string>();
                if ("all" == cores_str)
                    num_cores = get_number_of_default_cores(env);
                else
                    num_cores = hpx::util::safe_lexical_cast<std::size_t>(cores_str);
            }

            return num_cores;
        }
    }

    ///////////////////////////////////////////////////////////////////////
    bool command_line_handling::handle_arguments(util::manage_config& cfgmap,
        boost::program_options::variables_map& vm,
        std::vector<std::string>& ini_config, std::size_t& node)
    {
        using namespace boost::assign;

        bool debug_clp = node != std::size_t(-1) && vm.count("hpx:debug-clp");

        if (vm.count("hpx:ini")) {
            std::vector<std::string> cfg =
                vm["hpx:ini"].as<std::vector<std::string> >();
            std::copy(cfg.begin(), cfg.end(), std::back_inserter(ini_config));
            cfgmap.add(cfg);
        }

        // create host name mapping
        util::map_hostnames mapnames(debug_clp);

        if (vm.count("hpx:ifsuffix"))
            mapnames.use_suffix(vm["hpx:ifsuffix"].as<std::string>());
        if (vm.count("hpx:ifprefix"))
            mapnames.use_prefix(vm["hpx:ifprefix"].as<std::string>());

        // The AGAS host name and port number are pre-initialized from
        //the command line
        std::string agas_host =
            cfgmap.get_value<std::string>("hpx.agas.address",
                rtcfg_.get_entry("hpx.agas.address", ""));
        boost::uint16_t agas_port =
            cfgmap.get_value<boost::uint16_t>("hpx.agas.port",
                boost::lexical_cast<boost::uint16_t>(
                    rtcfg_.get_entry("hpx.agas.port", HPX_INITIAL_IP_PORT)
                ));

        if (vm.count("hpx:agas")) {
            if (!util::split_ip_address(
                vm["hpx:agas"].as<std::string>(), agas_host, agas_port))
            {
                std::cerr
                    << "hpx::init: command line warning: illegal port "
                        "number given, using default value instead."
                    << std::endl;
            }
        }

        // Check command line arguments.

        if (vm.count("hpx:iftransform")) {
            util::sed_transform iftransform(vm["hpx:iftransform"].as<std::string>());

            // Check for parsing failures
            if (!iftransform) {
                throw hpx::detail::command_line_error(boost::str(boost::format(
                    "Could not parse --hpx:iftransform argument '%1%'") %
                    vm["hpx:iftransform"].as<std::string>()));
            }

            typedef util::map_hostnames::transform_function_type
                transform_function_type;
            mapnames.use_transform(transform_function_type(iftransform));
        }

        bool using_nodelist = false;

        std::vector<std::string> nodelist;

        if(vm.count("hpx:nodefile"))
        {
            if (vm.count("hpx:nodes")) {
                throw hpx::detail::command_line_error(
                    "Ambiguous command line options. "
                    "Do not specify more than one of the --hpx:nodefile and "
                    "--hpx:nodes options at the same time.");
            }
            std::string node_file = vm["hpx:nodefile"].as<std::string>();
            ini_config += "hpx.nodefile!=" + node_file;
            std::ifstream ifs(node_file.c_str());
            if (ifs.is_open())
            {
                if (debug_clp)
                    std::cerr << "opened: " << node_file << std::endl;
                std::string line;
                while (std::getline(ifs, line)) {
                    if (!line.empty()) {
                        nodelist.push_back(line);
                    }
                }
            }
            else {
                if (debug_clp)
                    std::cerr << "failed opening: " << node_file << std::endl;

                // raise hard error if node file could not be opened
                throw hpx::detail::command_line_error(boost::str(boost::format(
                    "Could not open nodefile: '%s'") % node_file));
            }
        }
        else if (vm.count("hpx:nodes")) {
            nodelist = vm["hpx:nodes"].as<std::vector<std::string> >();
        }

        bool enable_batch_env = vm.count("hpx:ignore-batch-env") == 0;
        util::batch_environment env(nodelist, rtcfg_, debug_clp, enable_batch_env);

        if(!nodelist.empty())
        {
            using_nodelist = true;
            ini_config += "hpx.nodes!=" + env.init_from_nodelist(
                nodelist, agas_host);
        }

        // let the batch environment decide about the AGAS host
        agas_host = env.agas_host_name(
            agas_host.empty() ? HPX_INITIAL_IP_ADDRESS : agas_host);

        // handle number of cores and threads
        num_threads_ = detail::handle_num_threads(cfgmap, vm, env, using_nodelist);
        num_cores_ = detail::handle_num_cores(cfgmap, vm, num_threads_, env);

        // handling number of localities, those might have already been initialized
        // from MPI environment
        num_localities_ = detail::handle_num_localities(cfgmap, vm, env,
            using_nodelist, num_localities_);

        // Determine our network port, use arbitrary port if running on one
        // locality.
        std::string hpx_host =
            cfgmap.get_value<std::string>("hpx.parcel.address",
                env.host_name(
                    rtcfg_.get_entry("hpx.parcel.address", HPX_INITIAL_IP_ADDRESS)
                ));

        // we expect dynamic connections if:
        //  - --hpx:expect-connecting-localities or
        //  - hpx.expect_connecting_localities=1 is given, or
        //  - num_localities > 1
        bool expect_connections =
            cfgmap.get_value<int>("hpx.expect_connecting_localities",
                num_localities_ > 1 ? 0 : 1) ? true : false;

        if (vm.count("hpx:expect-connecting-localities"))
            expect_connections = true;

        ini_config += std::string("hpx.expect_connecting_localities=") +
            (expect_connections ? "1" : "0");

        boost::uint16_t initial_hpx_port = 0;
        if (num_localities_ != 1 || expect_connections)
        {
            initial_hpx_port =
                boost::lexical_cast<boost::uint16_t>(
                    rtcfg_.get_entry("hpx.parcel.port", HPX_INITIAL_IP_PORT));
        }

        boost::uint16_t hpx_port =
            cfgmap.get_value<boost::uint16_t>("hpx.parcel.port", initial_hpx_port);

        bool run_agas_server = vm.count("hpx:run-agas-server") != 0;
        if (node == std::size_t(-1))
            node = env.retrieve_node_number();

        // If the user has not specified an explicit runtime mode we
        // retrieve it from the command line.
        if (hpx::runtime_mode_default == mode_) {
            // The default mode is console, i.e. all workers need to be
            // started with --worker/-w.
            mode_ = hpx::runtime_mode_console;
            if (vm.count("hpx:console") + vm.count("hpx:worker") +
                vm.count("hpx:connect") > 1)
            {
                throw hpx::detail::command_line_error(
                    "Ambiguous command line options. "
                    "Do not specify more than one of --hpx:console, "
                    "--hpx:worker, or --hpx:connect");
            }

            // In these cases we default to executing with an empty
            // hpx_main, except if specified otherwise.
            if (vm.count("hpx:worker")) {
                mode_ = hpx::runtime_mode_worker;

#if !defined(HPX_HAVE_RUN_MAIN_EVERYWHERE)
                // do not execute any explicit hpx_main except if asked
                // otherwise
                if (!vm.count("hpx:run-hpx-main") &&
                    !cfgmap.get_value<int>("hpx.run_hpx_main", 0))
                {
                    util::detail::reset_function(hpx_main_f_);
                }
#endif
            }
            else if (vm.count("hpx:connect")) {
                mode_ = hpx::runtime_mode_connect;
            }
        }

        // we initialize certain settings if --node is specified (or data
        // has been retrieved from the environment)
        if (mode_ == hpx::runtime_mode_connect) {
            // when connecting we need to select a unique port
            hpx_port = cfgmap.get_value<boost::uint16_t>("hpx.parcel.port",
                boost::lexical_cast<boost::uint16_t>(
                    rtcfg_.get_entry("hpx.parcel.port", HPX_CONNECTING_IP_PORT)
                ));

#if !defined(HPX_HAVE_RUN_MAIN_EVERYWHERE)
            // do not execute any explicit hpx_main except if asked
            // otherwise
            if (!vm.count("hpx:run-hpx-main") &&
                !cfgmap.get_value<int>("hpx.run_hpx_main", 0))
            {
                util::detail::reset_function(hpx_main_f_);
            }
#endif
        }
        else if (node != std::size_t(-1) || vm.count("hpx:node")) {
            // command line overwrites the environment
            if (vm.count("hpx:node")) {
                if (vm.count("hpx:agas")) {
                    throw hpx::detail::command_line_error(
                        "Command line option --hpx:node "
                        "is not compatible with --hpx:agas");
                }
                node = vm["hpx:node"].as<std::size_t>();
            }

            if (!vm.count("hpx:worker")) {
                if (env.agas_node() == node) {
                    // console node, by default runs AGAS
                    run_agas_server = true;
                    mode_ = hpx::runtime_mode_console;
                }
                else {
                    // don't use port zero for non-console localities
                    if (hpx_port == 0 && node != 0)
                        hpx_port = HPX_INITIAL_IP_PORT;

                    // each node gets an unique port
                    hpx_port = static_cast<boost::uint16_t>(hpx_port + node);
                    mode_ = hpx::runtime_mode_worker;

#if !defined(HPX_HAVE_RUN_MAIN_EVERYWHERE)
                    // do not execute any explicit hpx_main except if asked
                    // otherwise
                    if (!vm.count("hpx:run-hpx-main") &&
                        !cfgmap.get_value<int>("hpx.run_hpx_main", 0))
                    {
                        util::detail::reset_function(hpx_main_f_);
                    }
#endif
                }
            }

            // store node number in configuration, don't do that if we're on a
            // worker and the node number is zero
            if (!vm.count("hpx:worker") || node != 0)
            {
                ini_config += "hpx.locality!=" +
                    std::to_string(node);
            }
        }

        if (vm.count("hpx:hpx")) {
            if (!util::split_ip_address(vm["hpx:hpx"].as<std::string>(),
                    hpx_host, hpx_port))
            {
                std::cerr
                    << "hpx::init: command line warning: illegal port "
                        "number given, using default value instead."
                    << std::endl;
            }
        }

        if ((vm.count("hpx:connect") || mode_ == hpx::runtime_mode_connect) &&
            hpx_host == "127.0.0.1")
        {
            hpx_host = hpx::util::resolve_public_ip_address();
        }

        // handle setting related to schedulers
        queuing_ = detail::handle_queueing(cfgmap, vm, "local-priority");
        ini_config += "hpx.scheduler=" + queuing_;

        affinity_domain_ = detail::handle_affinity(cfgmap, vm, "pu");
        ini_config += "hpx.affinity=" + affinity_domain_;

        affinity_bind_ = detail::handle_affinity_bind(cfgmap, vm, "");
        if (!affinity_bind_.empty())
            ini_config += "hpx.bind!=" + affinity_bind_;

        pu_step_ = detail::handle_pu_step(cfgmap, vm, 1);
        ini_config += "hpx.pu_step=" + std::to_string(pu_step_);

        pu_offset_ = detail::handle_pu_offset(cfgmap, vm, 0);
        ini_config += "hpx.pu_offset=" + std::to_string(pu_offset_);

        numa_sensitive_ = detail::handle_numa_sensitive(cfgmap, vm,
            affinity_bind_.empty() ? 0 : 1);
        ini_config += "hpx.numa_sensitive=" + std::to_string(numa_sensitive_);

        // map host names to ip addresses, if requested
        hpx_host = mapnames.map(hpx_host, hpx_port);
        agas_host = mapnames.map(agas_host, agas_port);

        // sanity checks
        if (num_localities_ == 1 && !vm.count("hpx:agas") && !vm.count("hpx:node"))
        {
            // We assume we have to run the AGAS server if the number of
            // localities to run on is not specified (or is '1')
            // and no additional option (--hpx:agas or --hpx:node) has been
            // specified. That simplifies running small standalone
            // applications on one locality.
            run_agas_server = mode_ != runtime_mode_connect;
        }

        if (hpx_host == agas_host && hpx_port == agas_port) {
            // we assume that we need to run the agas server if the user
            // asked for the same network addresses for HPX and AGAS
            run_agas_server = mode_ != runtime_mode_connect;
        }
        else if (run_agas_server) {
            // otherwise, if the user instructed us to run the AGAS server,
            // we set the AGAS network address to the same value as the HPX
            // network address
            agas_host = hpx_host;
            agas_port = hpx_port;
        }
        else if (env.found_batch_environment()) {
            // in batch mode, if the network addresses are different and we
            // should not run the AGAS server we assume to be in worker mode
            mode_ = hpx::runtime_mode_worker;

#if !defined(HPX_HAVE_RUN_MAIN_EVERYWHERE)
            // do not execute any explicit hpx_main except if asked
            // otherwise
            if (!vm.count("hpx:run-hpx-main") &&
                !cfgmap.get_value<int>("hpx.run_hpx_main", 0))
            {
                util::detail::reset_function(hpx_main_f_);
            }
#endif
        }

        // write HPX and AGAS network parameters to the proper ini-file entries
        ini_config += "hpx.parcel.address=" + hpx_host;
        ini_config += "hpx.parcel.port=" + std::to_string(hpx_port);
        ini_config += "hpx.agas.address=" + agas_host;
        ini_config += "hpx.agas.port=" + std::to_string(agas_port);

        if (run_agas_server) {
            ini_config += "hpx.agas.service_mode=bootstrap";
            if (vm.count("hpx:run-agas-server-only"))
                ini_config += "hpx.components.load_external=0";
        }
        else if (vm.count("hpx:run-agas-server-only") &&
              !(env.found_batch_environment()))
        {
            throw hpx::detail::command_line_error(
                "Command line option --hpx:run-agas-server-only "
                "can be specified only for the node running the AGAS server.");
        }

        if (1 == num_localities_ && vm.count("hpx:run-agas-server-only")) {
            std::cerr << "hpx::init: command line warning: --hpx:run-agas-server-only "
                "used for single locality execution, application might "
                "not run properly." << std::endl;
        }

        // we can't run the AGAS server while connecting
        if (run_agas_server && mode_ == runtime_mode_connect) {
            throw hpx::detail::command_line_error(
                "Command line option error: can't run AGAS server"
                "while connecting to a running application.");
        }

        // Set whether the AGAS server is running as a dedicated runtime.
        // This decides whether the AGAS actions are executed with normal
        // priority (if dedicated) or with high priority (non-dedicated)
        if (vm.count("hpx:run-agas-server-only"))
            ini_config += "hpx.agas.dedicated_server=1";

        if (vm.count("hpx:debug-hpx-log")) {
            ini_config += "hpx.logging.console.destination=" +
                detail::convert_to_log_file(
                    vm["hpx:debug-hpx-log"].as<std::string>());
            ini_config += "hpx.logging.destination=" +
                detail::convert_to_log_file(
                    vm["hpx:debug-hpx-log"].as<std::string>());
            ini_config += "hpx.logging.console.level=5";
            ini_config += "hpx.logging.level=5";
        }

        if (vm.count("hpx:debug-agas-log")) {
            ini_config += "hpx.logging.console.agas.destination=" +
                detail::convert_to_log_file(
                    vm["hpx:debug-agas-log"].as<std::string>());
            ini_config += "hpx.logging.agas.destination=" +
                detail::convert_to_log_file(
                    vm["hpx:debug-agas-log"].as<std::string>());
            ini_config += "hpx.logging.console.agas.level=5";
            ini_config += "hpx.logging.agas.level=5";
        }

        if (vm.count("hpx:debug-parcel-log")) {
            ini_config += "hpx.logging.console.parcel.destination=" +
                detail::convert_to_log_file(
                    vm["hpx:debug-parcel-log"].as<std::string>());
            ini_config += "hpx.logging.parcel.destination=" +
                detail::convert_to_log_file(
                    vm["hpx:debug-parcel-log"].as<std::string>());
            ini_config += "hpx.logging.console.parcel.level=5";
            ini_config += "hpx.logging.parcel.level=5";
        }

        // Set number of cores and OS threads in configuration.
        ini_config += "hpx.os_threads=" +
            std::to_string(num_threads_);
        ini_config += "hpx.cores=" +
            std::to_string(num_cores_);

        // Set number of localities in configuration (do it everywhere,
        // even if this information is only used by the AGAS server).
        ini_config += "hpx.localities=" +
            std::to_string(num_localities_);

        // FIXME: AGAS V2: if a locality is supposed to run the AGAS
        //        service only and requests to use 'priority_local' as the
        //        scheduler, switch to the 'local' scheduler instead.
        ini_config += std::string("hpx.runtime_mode=") +
            get_runtime_mode_name(mode_);

        bool noshutdown_evaluate = false;
        if (vm.count("hpx:print-counter-at")) {
            std::vector<std::string> print_counters_at =
                vm["hpx:print-counter-at"].as<std::vector<std::string> >();

            for (std::string const& s: print_counters_at)
            {
                if (0 == std::string("startup").find(s))
                {
                    ini_config += "hpx.print_counter.startup!=1";
                    continue;
                }
                if (0 == std::string("shutdown").find(s))
                {
                    ini_config += "hpx.print_counter.shutdown!=1";
                    continue;
                }
                if (0 == std::string("noshutdown").find(s))
                {
                    ini_config += "hpx.print_counter.shutdown!=0";
                    noshutdown_evaluate = true;
                    continue;
                }

                throw hpx::detail::command_line_error(boost::str(boost::format(
                    "Invalid argument for option --hpx:print-counter-at: "
                    "'%1%', allowed values: 'startup', 'shutdown' (default), "
                    "'noshutdown'") % s));
            }
        }

        // if any counters have to be evaluated, always print at the end
        if (vm.count("hpx:print-counter"))
        {
            if (!noshutdown_evaluate)
                ini_config += "hpx.print_counter.shutdown!=1";
            if (vm.count("hpx:reset-counters"))
                ini_config += "hpx.print_counter.reset!=1";
        }

        if (debug_clp) {
            std::cerr << "Configuration before runtime start:\n";
            std::cerr << "-----------------------------------\n";
            for (std::string const& s : ini_config) {
                std::cerr << s << std::endl;
            }
            std::cerr << "-----------------------------------\n";
        }

        return true;
    }

    ///////////////////////////////////////////////////////////////////////////
    void command_line_handling::store_command_line(int argc, char** argv)
    {
        using namespace boost::assign;

        // Collect the command line for diagnostic purposes.
        std::string cmd_line;
        for (int i = 0; i < argc; ++i)
        {
            // quote only if it contains whitespace
            std::string arg(argv[i]); //-V108
            cmd_line += detail::enquote(arg);

            if ((i + 1) != argc)
                cmd_line += " ";
        }

        // Store the program name and the command line.
        ini_config_ += "hpx.cmd_line!=" + cmd_line;
    }

    ///////////////////////////////////////////////////////////////////////////
    void command_line_handling::store_unregistered_options(
        std::string const& cmd_name,
        std::vector<std::string> const& unregistered_options)
    {
        using namespace boost::assign;

        std::string unregistered_options_cmd_line;

        if (!unregistered_options.empty()) {
            typedef std::vector<std::string>::const_iterator iterator_type;

            iterator_type  end = unregistered_options.end();
            for (iterator_type  it = unregistered_options.begin(); it != end; ++it)
                unregistered_options_cmd_line += " " + detail::enquote(*it);

            ini_config_ += "hpx.unknown_cmd_line!=" +
                detail::enquote(cmd_name) + unregistered_options_cmd_line;
        }

        ini_config_ += "hpx.program_name!=" + cmd_name;
        ini_config_ += "hpx.reconstructed_cmd_line!=" +
            detail::enquote(cmd_name) + " " +
            util::reconstruct_command_line(vm_) + " " +
            unregistered_options_cmd_line;
    }

    ///////////////////////////////////////////////////////////////////////////
    bool command_line_handling::handle_help_options(
        boost::program_options::options_description const& help)
    {
        using namespace boost::assign;

        if (vm_.count("hpx:help")) {
            std::string help_option(vm_["hpx:help"].as<std::string>());
            if (0 == std::string("minimal").find(help_option)) {
                // print static help only
                std::cout << help << std::endl;
                return true;
            }
            else if (0 == std::string("full").find(help_option)) {
                // defer printing help until after dynamic part has been
                // acquired
                std::ostringstream strm;
                strm << help << std::endl;
                ini_config_ += "hpx.cmd_line_help!=" +
                    detail::encode_string(strm.str());
                ini_config_ += "hpx.cmd_line_help_option!=" + help_option;
            }
            else {
                throw hpx::detail::command_line_error(boost::str(boost::format(
                    "Invalid argument for option --hpx:help: '%1%', allowed values: "
                    "'minimal' (default) and 'full'") % help_option));
            }
        }
        return false;
    }

    void attach_debugger()
    {
#if defined(_POSIX_VERSION)
        volatile int i = 0;
        std::cerr
            << "PID: " << getpid() << " on " << boost::asio::ip::host_name()
            << " ready for attaching debugger. Once attached set i = 1 and continue"
            << std::endl;
        while(i == 0)
        {
            sleep(1);
        }
#elif defined(HPX_WINDOWS)
        DebugBreak();
#endif
    }

    void command_line_handling::handle_attach_debugger()
    {
#if defined(_POSIX_VERSION) || defined(HPX_WINDOWS)
        if(vm_.count("hpx:attach-debugger"))
        {
            std::string option = vm_["hpx:attach-debugger"].as<std::string>();
            if (option != "startup" && option != "exception") {
                std::cerr <<
                    "hpx::init: command line warning: --hpx:attach-debugger: "
                    "invalid option: " << option << ". Allowed values are "
                    "'startup' or 'exception'" << std::endl;
            }
            else {
                if (option == "startup")
                    attach_debugger();

                using namespace boost::assign;
                ini_config_ += "hpx.attach_debugger!=" + option;
            }
        }
#endif
    }

#if defined(HPX_HAVE_HWLOC)
    ///////////////////////////////////////////////////////////////////////////
    void handle_print_bind(boost::program_options::variables_map const& vm_,
        std::size_t num_threads)
    {
        threads::topology& top = threads::create_topology();
        runtime & rt = get_runtime();
        {
            std::ostringstream strm;    // make sure all output is kept together

            strm << std::string(79, '*') << '\n';
            strm << "locality: " << hpx::get_locality_id() << '\n';
            for (std::size_t i = 0; i != num_threads; ++i)
            {
                // print the mask for the current PU
                threads::mask_cref_type pu_mask =
                    rt.get_thread_manager().get_pu_mask(top, i);

                if (!threads::any(pu_mask))
                {
                    strm << std::setw(4) << i << ": thread binding disabled" //-V112
                         << std::endl;
                }
                else
                {
                    top.print_affinity_mask(strm, i, pu_mask);
                }

                // Make sure the mask does not contradict the CPU bindings
                // returned by the system (see #973: Would like option to
                // report HWLOC bindings).
                error_code ec(lightweight);
                threads::mask_type boundcpu = top.get_cpubind_mask(
                    rt.get_thread_manager().get_os_thread_handle(i), ec);

                // The masks reported by HPX must be the same as the ones
                // reported from HWLOC.
                if (!ec && threads::any(boundcpu) &&
                    !threads::equal(boundcpu, pu_mask, num_threads))
                {
                    HPX_THROW_EXCEPTION(invalid_status,
                        "handle_print_bind",
                        boost::str(
                            boost::format("unexpected mismatch between "
                                "locality %1%: binding reported from HWLOC(%2%) "
                                " and HPX(%3%) on thread %4%"
                            ) % hpx::get_locality_id() % boundcpu % pu_mask % i));
                }
            }

            std::cout << strm.str();
        }
    }
#endif

    void handle_list_parcelports()
    {
        runtime & rt = get_runtime();
        {
            std::ostringstream strm;    // make sure all output is kept together
            strm << std::string(79, '*') << '\n';
            strm << "locality: " << hpx::get_locality_id() << '\n';

            rt.get_parcel_handler().list_parcelports(strm);

            std::cout << strm.str();
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    int command_line_handling::call(
        boost::program_options::options_description const& desc_cmdline,
        int argc, char** argv)
    {
        util::manage_config cfgmap(ini_config_);

        std::vector<boost::shared_ptr<plugins::plugin_registry_base> >
            plugin_registries = rtcfg_.load_modules();

        // insert the pre-configured ini settings after loading modules
        for (std::string const& e : ini_config_)
            rtcfg_.parse("<user supplied config>", e, true, false);

        // Initial analysis of the command line options. This is
        // preliminary as it will not take into account any aliases as
        // defined in any of the runtime configuration files.
        {
            // Boost V1.47 and before do not properly reset a variables_map
            // when calling vm.clear(). We work around that problems by
            // creating a separate instance just for the preliminary
            // command line handling.
            boost::program_options::variables_map prevm;
            if (!util::parse_commandline(rtcfg_, desc_cmdline, argc,
                    argv, prevm, std::size_t(-1), util::allow_unregistered,
                    mode_))
            {
                return -1;
            }

            // handle all --hpx:foo options, determine node
            std::vector<std::string> ini_config;    // will be discarded
            if (!handle_arguments(cfgmap, prevm, ini_config, node_))
                return -2;

            // re-initialize runtime configuration object
            if (prevm.count("hpx:config"))
                rtcfg_.reconfigure(prevm["hpx:config"].as<std::string>());
            else
                rtcfg_.reconfigure("");

            // Make sure any aliases defined on the command line get used
            // for the option analysis below.
            std::vector<std::string> cfg;
            if (prevm.count("hpx:ini")) {
                cfg = prevm["hpx:ini"].as<std::vector<std::string> >();
                cfgmap.add(cfg);
            }

            // append ini options from command line
            std::copy(ini_config_.begin(), ini_config_.end(),
                std::back_inserter(cfg));

            rtcfg_.reconfigure(cfg);
        }

        // Re-run program option analysis, ini settings (such as aliases)
        // will be considered now.

        parcelset::parcelhandler::init(&argc, &argv, *this);
        for (boost::shared_ptr<plugins::plugin_registry_base>& reg : plugin_registries)
        {
            reg->init(&argc, &argv, *this);
        }

        // minimally assume one locality and this is the console
        if (node_ == std::size_t(-1))
            node_ = 0;

        // Now re-parse the command line using the node number (if given).
        // This will additionally detect any --hpx:N:foo options.
        boost::program_options::options_description help;
        std::vector<std::string> unregistered_options;

        if (!util::parse_commandline(rtcfg_, desc_cmdline,
                argc, argv, vm_, node_, util::allow_unregistered, mode_,
                &help, &unregistered_options))
        {
            return -1;
        }

        // break into debugger, if requested
        handle_attach_debugger();

        // handle all --hpx:foo and --hpx:*:foo options
        if (!handle_arguments(cfgmap, vm_, ini_config_, node_))
            return -2;

        // store unregistered command line and arguments
        store_command_line(argc, argv);
        store_unregistered_options(argv[0], unregistered_options);

        // help can be printed only after the runtime mode has been set
        if (handle_help_options(help))
            return 1;     // exit application gracefully

        // add all remaining ini settings to the global configuration
        rtcfg_.reconfigure(ini_config_);

        // print version/copyright information
        if (vm_.count("hpx:version")) {
            detail::print_version(std::cout);
            return 1;
        }

        // print configuration information (static and dynamic)
        if (vm_.count("hpx:info")) {
            detail::print_info(std::cout, *this);
            return 1;
        }

        // all is good
        return 0;
    }
}}
