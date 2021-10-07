//  Copyright (c) 2007-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/local/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/command_line_handling_local/command_line_handling_local.hpp>
#include <hpx/command_line_handling_local/parse_command_line_local.hpp>
#include <hpx/functional/detail/reset_function.hpp>
#include <hpx/local/version.hpp>
#include <hpx/modules/asio.hpp>
#include <hpx/modules/debugging.hpp>
#include <hpx/modules/format.hpp>
#include <hpx/modules/program_options.hpp>
#include <hpx/modules/runtime_configuration.hpp>
#include <hpx/modules/topology.hpp>
#include <hpx/modules/util.hpp>
#include <hpx/preprocessor/stringize.hpp>
#include <hpx/type_support/unused.hpp>
#include <hpx/util/from_string.hpp>

#include <boost/tokenizer.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace hpx { namespace local { namespace detail {
    std::string runtime_configuration_string(command_line_handling const& cfg)
    {
        std::ostringstream strm;

        // default scheduler used for this run
        strm << "  {scheduler}: " << cfg.queuing_ << "\n";

        // amount of threads and cores configured for this run
        strm << "  {os-threads}: " << cfg.num_threads_ << "\n";
        strm << "  {cores}: " << cfg.num_cores_ << "\n";

        return strm.str();
    }

    ///////////////////////////////////////////////////////////////////////
    int print_version(std::ostream& out)
    {
        out << std::endl << hpx::local::copyright() << std::endl;
        out << hpx::local::complete_version() << std::endl;
        return 1;
    }

    int print_info(std::ostream& out, command_line_handling const& cfg)
    {
        out << "Static configuration:\n---------------------\n";
        out << hpx::local::configuration_string() << std::endl;

        out << "Runtime configuration:\n----------------------\n";
        out << runtime_configuration_string(cfg) << std::endl;

        return 1;
    }

    ///////////////////////////////////////////////////////////////////////
    inline void encode(
        std::string& str, char s, char const* r, std::size_t inc = 1ull)
    {
        std::string::size_type pos = 0;
        while ((pos = str.find_first_of(s, pos)) != std::string::npos)
        {
            str.replace(pos, 1, r);
            pos += inc;
        }
    }

    inline std::string encode_string(std::string str)
    {
        encode(str, '\n', "\\n");
        return str;
    }

    inline std::string encode_and_enquote(std::string str)
    {
        encode(str, '\"', "\\\"", 2);
        return util::detail::enquote(std::move(str));
    }

    ///////////////////////////////////////////////////////////////////////
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

    std::string handle_queuing(util::manage_config& cfgmap,
        hpx::program_options::variables_map& vm, std::string const& default_)
    {
        // command line options is used preferred
        if (vm.count("hpx:queuing"))
            return vm["hpx:queuing"].as<std::string>();

        // use either cfgmap value or default
        return cfgmap.get_value<std::string>("hpx.scheduler", default_);
    }

    std::string handle_affinity(util::manage_config& cfgmap,
        hpx::program_options::variables_map& vm, std::string const& default_)
    {
        // command line options is used preferred
        if (vm.count("hpx:affinity"))
            return vm["hpx:affinity"].as<std::string>();

        // use either cfgmap value or default
        return cfgmap.get_value<std::string>("hpx.affinity", default_);
    }

    std::string handle_affinity_bind(util::manage_config& cfgmap,
        hpx::program_options::variables_map& vm, std::string const& default_)
    {
        // command line options is used preferred
        if (vm.count("hpx:bind"))
        {
            std::string affinity_desc;

            std::vector<std::string> bind_affinity =
                vm["hpx:bind"].as<std::vector<std::string>>();
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
        hpx::program_options::variables_map& vm, std::size_t default_)
    {
        // command line options is used preferred
        if (vm.count("hpx:pu-step"))
            return vm["hpx:pu-step"].as<std::size_t>();

        // use either cfgmap value or default
        return cfgmap.get_value<std::size_t>("hpx.pu_step", default_);
    }

    std::size_t handle_pu_offset(util::manage_config& cfgmap,
        hpx::program_options::variables_map& vm, std::size_t default_)
    {
        // command line options is used preferred
        if (vm.count("hpx:pu-offset"))
            return vm["hpx:pu-offset"].as<std::size_t>();

        // use either cfgmap value or default
        return cfgmap.get_value<std::size_t>("hpx.pu_offset", default_);
    }

    std::size_t handle_numa_sensitive(util::manage_config& cfgmap,
        hpx::program_options::variables_map& vm, std::size_t default_)
    {
        if (vm.count("hpx:numa-sensitive") != 0)
        {
            std::size_t numa_sensitive =
                vm["hpx:numa-sensitive"].as<std::size_t>();
            if (numa_sensitive > 2)
            {
                throw hpx::detail::command_line_error(
                    "Invalid argument "
                    "value for --hpx:numa-sensitive. Allowed values "
                    "are "
                    "0, 1, or 2");
            }
            return numa_sensitive;
        }

        // use either cfgmap value or default
        return cfgmap.get_value<std::size_t>("hpx.numa_sensitive", default_);
    }

    ///////////////////////////////////////////////////////////////////////
    std::size_t get_number_of_default_threads(bool use_process_mask)
    {
        if (use_process_mask)
        {
            threads::topology& top = threads::create_topology();
            return threads::count(top.get_cpubind_mask());
        }
        else
        {
            return threads::hardware_concurrency();
        }
    }

    std::size_t get_number_of_default_cores(bool use_process_mask)
    {
        threads::topology& top = threads::create_topology();

        std::size_t num_cores = top.get_number_of_cores();

        if (use_process_mask)
        {
            threads::mask_type proc_mask = top.get_cpubind_mask();
            std::size_t num_cores_proc_mask = 0;

            for (std::size_t num_core = 0; num_core < num_cores; ++num_core)
            {
                threads::mask_type core_mask =
                    top.init_core_affinity_mask_from_core(num_core);
                if (threads::bit_and(core_mask, proc_mask))
                {
                    ++num_cores_proc_mask;
                }
            }

            return num_cores_proc_mask;
        }

        return num_cores;
    }

    ///////////////////////////////////////////////////////////////////////
    std::size_t handle_num_threads(util::manage_config& cfgmap,
        hpx::util::runtime_configuration const& rtcfg,
        hpx::program_options::variables_map& vm, bool use_process_mask)
    {
        // If using the process mask we override "cores" and "all" options but
        // keep explicit numeric values.
        const std::size_t init_threads =
            get_number_of_default_threads(use_process_mask);
        const std::size_t init_cores =
            get_number_of_default_cores(use_process_mask);

        std::size_t default_threads = init_threads;

        std::string threads_str = cfgmap.get_value<std::string>(
            "hpx.os_threads",
            rtcfg.get_entry("hpx.os_threads", std::to_string(default_threads)));

        if ("cores" == threads_str)
        {
            default_threads = init_cores;
        }
        else if ("all" == threads_str)
        {
            default_threads = init_threads;
        }
        else
        {
            default_threads = hpx::util::from_string<std::size_t>(threads_str);
        }

        std::size_t threads =
            cfgmap.get_value<std::size_t>("hpx.os_threads", default_threads);

        if (vm.count("hpx:threads"))
        {
            threads_str = vm["hpx:threads"].as<std::string>();
            if ("all" == threads_str)
            {
                threads = init_threads;
            }
            else if ("cores" == threads_str)
            {
                threads = init_cores;
            }
            else
            {
                threads = hpx::util::from_string<std::size_t>(threads_str);
            }

            if (threads == 0)
            {
                throw hpx::detail::command_line_error(
                    "Number of --hpx:threads must be greater than 0");
            }

#if defined(HPX_HAVE_MAX_CPU_COUNT)
            if (threads > HPX_HAVE_MAX_CPU_COUNT)
            {
                // clang-format off
                    throw hpx::detail::command_line_error("Requested more than "
                        HPX_PP_STRINGIZE(HPX_HAVE_MAX_CPU_COUNT)" --hpx:threads "
                        "to use for this application, use the option "
                        "-DHPXLocal_WITH_MAX_CPU_COUNT=<N> when configuring HPX.");
                // clang-format on
            }
#endif
        }

        // make sure minimal requested number of threads is observed
        std::size_t min_os_threads =
            cfgmap.get_value<std::size_t>("hpx.force_min_os_threads", threads);

        if (min_os_threads == 0)
        {
            throw hpx::detail::command_line_error(
                "Number of hpx.force_min_os_threads must be greater "
                "than "
                "0");
        }

#if defined(HPX_HAVE_MAX_CPU_COUNT)
        if (min_os_threads > HPX_HAVE_MAX_CPU_COUNT)
        {
            throw hpx::detail::command_line_error(
                "Requested more than " HPX_PP_STRINGIZE(
                    HPX_HAVE_MAX_CPU_COUNT) " hpx.force_min_os_threads "
                                            "to use for this "
                                            "application, "
                                            "use the option "
                                            "-DHPXLocal_WITH_MAX_CPU_COUNT=<"
                                            "N> "
                                            "when configuring HPX.");
        }
#endif

        threads = (std::max)(threads, min_os_threads);

        return threads;
    }

    std::size_t handle_num_cores(util::manage_config& cfgmap,
        hpx::program_options::variables_map& vm, std::size_t num_threads,
        bool use_process_mask)
    {
        std::string cores_str = cfgmap.get_value<std::string>("hpx.cores", "");
        if ("all" == cores_str)
        {
            cfgmap.config_["hpx.cores"] =
                std::to_string(get_number_of_default_cores(use_process_mask));
        }

        std::size_t num_cores =
            cfgmap.get_value<std::size_t>("hpx.cores", num_threads);
        if (vm.count("hpx:cores"))
        {
            cores_str = vm["hpx:cores"].as<std::string>();
            if ("all" == cores_str)
            {
                num_cores = get_number_of_default_cores(use_process_mask);
            }
            else
            {
                num_cores = hpx::util::from_string<std::size_t>(cores_str);
            }
        }

        return num_cores;
    }

    ///////////////////////////////////////////////////////////////////////
    void command_line_handling::check_affinity_domain() const
    {
        if (affinity_domain_ != "pu")
        {
            if (0 != std::string("pu").find(affinity_domain_) &&
                0 != std::string("core").find(affinity_domain_) &&
                0 != std::string("numa").find(affinity_domain_) &&
                0 != std::string("machine").find(affinity_domain_))
            {
                throw hpx::detail::command_line_error(
                    "Invalid command line option "
                    "--hpx:affinity, value must be one of: pu, core, "
                    "numa, "
                    "or machine.");
            }
        }
    }

    void command_line_handling::check_affinity_description() const
    {
        if (affinity_bind_.empty())
        {
            return;
        }

        if (!(pu_offset_ == std::size_t(-1) || pu_offset_ == std::size_t(0)) ||
            pu_step_ != 1 || affinity_domain_ != "pu")
        {
            throw hpx::detail::command_line_error(
                "Command line option --hpx:bind "
                "should not be used with --hpx:pu-step, "
                "--hpx:pu-offset, "
                "or --hpx:affinity.");
        }
    }

    void command_line_handling::check_pu_offset() const
    {
        if (pu_offset_ != std::size_t(-1) &&
            pu_offset_ >= hpx::threads::hardware_concurrency())
        {
            throw hpx::detail::command_line_error(
                "Invalid command line option "
                "--hpx:pu-offset, value must be smaller than number of "
                "available processing units.");
        }
    }

    void command_line_handling::check_pu_step() const
    {
        if (hpx::threads::hardware_concurrency() > 1 &&
            (pu_step_ == 0 || pu_step_ >= hpx::threads::hardware_concurrency()))
        {
            throw hpx::detail::command_line_error(
                "Invalid command line option "
                "--hpx:pu-step, value must be non-zero and smaller "
                "than "
                "number of available processing units.");
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    bool command_line_handling::handle_arguments(util::manage_config& cfgmap,
        hpx::program_options::variables_map& vm,
        std::vector<std::string>& ini_config)
    {
        bool debug_clp = vm.count("hpx:debug-clp");

        if (vm.count("hpx:ini"))
        {
            std::vector<std::string> cfg =
                vm["hpx:ini"].as<std::vector<std::string>>();
            std::copy(cfg.begin(), cfg.end(), std::back_inserter(ini_config));
            cfgmap.add(cfg);
        }

        use_process_mask_ =
            (cfgmap.get_value<int>("hpx.use_process_mask", 0) > 0) ||
            (vm.count("hpx:use-process-mask") > 0);

#if defined(__APPLE__)
        if (use_process_mask_)
        {
            std::cerr << "Warning: enabled process mask for thread "
                         "binding, but "
                         "thread binding is not supported on macOS. "
                         "Ignoring option."
                      << std::endl;
            use_process_mask_ = false;
        }
#endif

        ini_config.emplace_back(
            "hpx.use_process_mask!=" + std::to_string(use_process_mask_));

        // handle setting related to schedulers
        queuing_ = detail::handle_queuing(cfgmap, vm, "local-priority-fifo");
        ini_config.emplace_back("hpx.scheduler=" + queuing_);

        affinity_domain_ = detail::handle_affinity(cfgmap, vm, "pu");
        ini_config.emplace_back("hpx.affinity=" + affinity_domain_);

        check_affinity_domain();

        affinity_bind_ = detail::handle_affinity_bind(cfgmap, vm, "");
        if (!affinity_bind_.empty())
        {
#if defined(__APPLE__)
            std::cerr << "Warning: thread binding set to \"" << affinity_bind_
                      << "\" but thread binding is not supported on macOS. "
                         "Ignoring option."
                      << std::endl;
            affinity_bind_ = "";
#else
            ini_config.emplace_back("hpx.bind!=" + affinity_bind_);
#endif
        }

        pu_step_ = detail::handle_pu_step(cfgmap, vm, 1);
#if defined(__APPLE__)
        if (pu_step_ != 1)
        {
            std::cerr << "Warning: PU step set to \"" << pu_step_
                      << "\" but thread binding is not supported on macOS. "
                         "Ignoring option."
                      << std::endl;
            pu_step_ = 1;
        }
#endif
        ini_config.emplace_back("hpx.pu_step=" + std::to_string(pu_step_));

        check_pu_step();

        pu_offset_ = detail::handle_pu_offset(cfgmap, vm, std::size_t(-1));

        // NOLINTNEXTLINE(bugprone-branch-clone)
        if (pu_offset_ != std::size_t(-1))
        {
#if defined(__APPLE__)
            std::cerr << "Warning: PU offset set to \"" << pu_offset_
                      << "\" but thread binding is not supported on macOS. "
                         "Ignoring option."
                      << std::endl;
            pu_offset_ = std::size_t(-1);
            ini_config.emplace_back("hpx.pu_offset=0");
#else
            ini_config.emplace_back(
                "hpx.pu_offset=" + std::to_string(pu_offset_));
#endif
        }
        else
        {
            ini_config.emplace_back("hpx.pu_offset=0");
        }

        check_pu_offset();

        numa_sensitive_ = detail::handle_numa_sensitive(
            cfgmap, vm, affinity_bind_.empty() ? 0 : 1);
        ini_config.emplace_back(
            "hpx.numa_sensitive=" + std::to_string(numa_sensitive_));

        // default affinity mode is now 'balanced' (only if no pu-step or
        // pu-offset is given)
        if (pu_step_ == 1 && pu_offset_ == std::size_t(-1) &&
            affinity_bind_.empty())
        {
#if defined(__APPLE__)
            affinity_bind_ = "none";
#else
            affinity_bind_ = "balanced";
#endif
            ini_config.emplace_back("hpx.bind!=" + affinity_bind_);
        }

        check_affinity_description();

        // handle number of cores and threads
        num_threads_ =
            detail::handle_num_threads(cfgmap, rtcfg_, vm, use_process_mask_);
        num_cores_ = detail::handle_num_cores(
            cfgmap, vm, num_threads_, use_process_mask_);

        // Set number of cores and OS threads in configuration.
        ini_config.emplace_back(
            "hpx.os_threads=" + std::to_string(num_threads_));
        ini_config.emplace_back("hpx.cores=" + std::to_string(num_cores_));

        if (vm_.count("hpx:high-priority-threads"))
        {
            std::size_t num_high_priority_queues =
                vm_["hpx:high-priority-threads"].as<std::size_t>();
            if (num_high_priority_queues != std::size_t(-1) &&
                num_high_priority_queues > num_threads_)
            {
                throw hpx::detail::command_line_error(
                    "Invalid command line option: "
                    "number of high priority threads ("
                    "--hpx:high-priority-threads), should not be "
                    "larger "
                    "than number of threads (--hpx:threads)");
            }

            if (!(queuing_ == "local-priority" || queuing_ == "abp-priority"))
            {
                throw hpx::detail::command_line_error(
                    "Invalid command line option "
                    "--hpx:high-priority-threads, "
                    "valid for --hpx:queuing=local-priority and "
                    "--hpx:queuing=abp-priority only");
            }

            ini_config.emplace_back("hpx.thread_queue.high_priority_queues!=" +
                std::to_string(num_high_priority_queues));
        }

        enable_logging_settings(vm, ini_config);

        if (debug_clp)
        {
            std::cerr << "Configuration before runtime start:\n";
            std::cerr << "-----------------------------------\n";
            for (std::string const& s : ini_config)
            {
                std::cerr << s << std::endl;
            }
            std::cerr << "-----------------------------------\n";
        }

        return true;
    }

    ///////////////////////////////////////////////////////////////////////////
    void command_line_handling::enable_logging_settings(
        hpx::program_options::variables_map& vm,
        std::vector<std::string>& ini_config)
    {
#if defined(HPX_HAVE_LOGGING)
        if (vm.count("hpx:debug-hpx-log"))
        {
            ini_config.emplace_back("hpx.logging.console.destination=" +
                detail::convert_to_log_file(
                    vm["hpx:debug-hpx-log"].as<std::string>()));
            ini_config.emplace_back("hpx.logging.destination=" +
                detail::convert_to_log_file(
                    vm["hpx:debug-hpx-log"].as<std::string>()));
            ini_config.emplace_back("hpx.logging.console.level=5");
            ini_config.emplace_back("hpx.logging.level=5");
        }

        if (vm.count("hpx:debug-agas-log"))
        {
            ini_config.emplace_back("hpx.logging.console.agas.destination=" +
                detail::convert_to_log_file(
                    vm["hpx:debug-agas-log"].as<std::string>()));
            ini_config.emplace_back("hpx.logging.agas.destination=" +
                detail::convert_to_log_file(
                    vm["hpx:debug-agas-log"].as<std::string>()));
            ini_config.emplace_back("hpx.logging.console.agas.level=5");
            ini_config.emplace_back("hpx.logging.agas.level=5");
        }

        if (vm.count("hpx:debug-parcel-log"))
        {
            ini_config.emplace_back("hpx.logging.console.parcel.destination=" +
                detail::convert_to_log_file(
                    vm["hpx:debug-parcel-log"].as<std::string>()));
            ini_config.emplace_back("hpx.logging.parcel.destination=" +
                detail::convert_to_log_file(
                    vm["hpx:debug-parcel-log"].as<std::string>()));
            ini_config.emplace_back("hpx.logging.console.parcel.level=5");
            ini_config.emplace_back("hpx.logging.parcel.level=5");
        }

        if (vm.count("hpx:debug-timing-log"))
        {
            ini_config.emplace_back("hpx.logging.console.timing.destination=" +
                detail::convert_to_log_file(
                    vm["hpx:debug-timing-log"].as<std::string>()));
            ini_config.emplace_back("hpx.logging.timing.destination=" +
                detail::convert_to_log_file(
                    vm["hpx:debug-timing-log"].as<std::string>()));
            ini_config.emplace_back("hpx.logging.console.timing.level=1");
            ini_config.emplace_back("hpx.logging.timing.level=1");
        }

        if (vm.count("hpx:debug-app-log"))
        {
            ini_config.emplace_back(
                "hpx.logging.console.application.destination=" +
                detail::convert_to_log_file(
                    vm["hpx:debug-app-log"].as<std::string>()));
            ini_config.emplace_back("hpx.logging.application.destination=" +
                detail::convert_to_log_file(
                    vm["hpx:debug-app-log"].as<std::string>()));
            ini_config.emplace_back("hpx.logging.console.application.level=5");
            ini_config.emplace_back("hpx.logging.application.level=5");
        }
#else
        if (vm.count("hpx:debug-hpx-log") || vm.count("hpx:debug-agas-log") ||
            vm.count("hpx:debug-parcel-log") ||
            vm.count("hpx:debug-timing-log") || vm.count("hpx:debug-app-log"))
        {
            // clang-format off
            throw hpx::detail::command_line_error(
                "Command line option error: can't enable logging while it "
                "was disabled at configuration time. Please re-configure "
                "HPX using the option -DHPXLocal_WITH_LOGGING=On.");
            // clang-format on
        }

        HPX_UNUSED(ini_config);
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    void command_line_handling::store_command_line(int argc, char** argv)
    {
        // Collect the command line for diagnostic purposes.
        std::string command;
        std::string cmd_line;
        std::string options;
        for (int i = 0; i < argc; ++i)
        {
            // quote only if it contains whitespace
            std::string arg = detail::encode_and_enquote(argv[i]);    //-V108

            cmd_line += arg;
            if (i == 0)
            {
                command = arg;
            }
            else
            {
                options += " " + arg;
            }

            if ((i + 1) != argc)
            {
                cmd_line += " ";
            }
        }

        // Store the program name and the command line.
        ini_config_.emplace_back("hpx.cmd_line!=" + cmd_line);
        ini_config_.emplace_back("hpx.commandline.command!=" + command);
        ini_config_.emplace_back("hpx.commandline.options!=" + options);
    }

    ///////////////////////////////////////////////////////////////////////////
    void command_line_handling::store_unregistered_options(
        std::string const& cmd_name,
        std::vector<std::string> const& unregistered_options)
    {
        std::string unregistered_options_cmd_line;

        if (!unregistered_options.empty())
        {
            typedef std::vector<std::string>::const_iterator iterator_type;

            iterator_type end = unregistered_options.end();
            for (iterator_type it = unregistered_options.begin(); it != end;
                 ++it)
                unregistered_options_cmd_line +=
                    " " + detail::encode_and_enquote(*it);

            ini_config_.emplace_back("hpx.unknown_cmd_line!=" +
                detail::encode_and_enquote(cmd_name) +
                unregistered_options_cmd_line);
        }

        ini_config_.emplace_back("hpx.program_name!=" + cmd_name);
        ini_config_.emplace_back("hpx.reconstructed_cmd_line!=" +
            encode_and_enquote(cmd_name) + " " + reconstruct_command_line(vm_) +
            " " + unregistered_options_cmd_line);
    }

    ///////////////////////////////////////////////////////////////////////////
    bool command_line_handling::handle_help_options(
        hpx::program_options::options_description const& help)
    {
        if (vm_.count("hpx:help"))
        {
            std::string help_option(vm_["hpx:help"].as<std::string>());
            if (0 == std::string("minimal").find(help_option))
            {
                // print static help only
                std::cout << help << std::endl;
                return true;
            }
            else if (0 == std::string("full").find(help_option))
            {
                // defer printing help until after dynamic part has been
                // acquired
                std::ostringstream strm;
                strm << help << std::endl;
                ini_config_.emplace_back(
                    "hpx.cmd_line_help!=" + detail::encode_string(strm.str()));
                ini_config_.emplace_back(
                    "hpx.cmd_line_help_option!=" + help_option);
            }
            else
            {
                throw hpx::detail::command_line_error(
                    hpx::util::format("Invalid argument for option --hpx:help: "
                                      "'{1}', allowed values: "
                                      "'minimal' (default) and 'full'",
                        help_option));
            }
        }
        return false;
    }

    void command_line_handling::handle_attach_debugger()
    {
#if defined(_POSIX_VERSION) || defined(HPX_WINDOWS)
        if (vm_.count("hpx:attach-debugger"))
        {
            std::string option = vm_["hpx:attach-debugger"].as<std::string>();
            if (option != "off" && option != "startup" &&
                option != "exception" && option != "test-failure")
            {
                // clang-format off
                std::cerr <<
                    "hpx::init: command line warning: --hpx:attach-debugger: "
                    "invalid option: " << option << ". Allowed values are "
                    "'off', 'startup', 'exception' or 'test-failure'" << std::endl;
                // clang-format on
            }
            else
            {
                if (option == "startup")
                    util::attach_debugger();

                ini_config_.emplace_back("hpx.attach_debugger!=" + option);
            }
        }
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    // separate command line arguments from configuration settings
    std::vector<std::string> command_line_handling::preprocess_config_settings(
        int argc, char** argv)
    {
        std::vector<std::string> options;
        options.reserve(static_cast<std::size_t>(argc) + ini_config_.size());

        // extract all command line arguments from configuration settings and
        // remove them from this list
        auto it = std::stable_partition(ini_config_.begin(), ini_config_.end(),
            [](std::string const& e) { return e.find("--hpx:") != 0; });

        std::move(it, ini_config_.end(), std::back_inserter(options));
        ini_config_.erase(it, ini_config_.end());

        // store the command line options that came from the configuration
        // settings in the registry
        if (!options.empty())
        {
            std::string config_options;
            for (auto const& option : options)
            {
                config_options += " " + option;
            }

            rtcfg_.add_entry("hpx.commandline.config_options", config_options);
        }

        // now append all original command line options
        for (int i = 1; i != argc; ++i)
        {
            options.emplace_back(argv[i]);
        }

        return options;
    }

    ///////////////////////////////////////////////////////////////////////////
    std::vector<std::string> prepend_options(
        std::vector<std::string>&& args, std::string&& options)
    {
        if (options.empty())
        {
            return std::move(args);
        }

        using tokenizer = boost::tokenizer<boost::escaped_list_separator<char>>;
        boost::escaped_list_separator<char> sep('\\', ' ', '\"');
        tokenizer tok(options, sep);

        std::vector<std::string> result(tok.begin(), tok.end());
        std::move(args.begin(), args.end(), std::back_inserter(result));
        return result;
    }

    ///////////////////////////////////////////////////////////////////////////
    int command_line_handling::call(
        hpx::program_options::options_description const& desc_cmdline, int argc,
        char** argv)
    {
        // set the flag signaling that command line parsing has been done
        cmd_line_parsed_ = true;

        // separate command line arguments from configuration settings
        std::vector<std::string> args = preprocess_config_settings(argc, argv);

        util::manage_config cfgmap(ini_config_);

        // insert the pre-configured ini settings before loading modules
        for (std::string const& e : ini_config_)
            rtcfg_.parse("<user supplied config>", e, true, false);

        // support re-throwing command line exceptions for testing purposes
        int error_mode = util::allow_unregistered;
        if (cfgmap.get_value("hpx.commandline.rethrow_errors", 0) != 0)
        {
            error_mode |= util::rethrow_on_error;
        }

        // The cfg registry may hold command line options to prepend to the
        // real command line.
        std::string prepend_command_line =
            rtcfg_.get_entry("hpx.commandline.prepend_options");

        args =
            prepend_options(std::move(args), std::move(prepend_command_line));

        // Initial analysis of the command line options. This is
        // preliminary as it will not take into account any aliases as
        // defined in any of the runtime configuration files.
        {
            // Boost V1.47 and before do not properly reset a variables_map
            // when calling vm.clear(). We work around that problems by
            // creating a separate instance just for the preliminary
            // command line handling.
            hpx::program_options::variables_map prevm;
            if (!parse_commandline(
                    rtcfg_, desc_cmdline, argv[0], args, prevm, error_mode))
            {
                return -1;
            }

            // handle all --hpx:foo options
            std::vector<std::string> ini_config;    // discard
            if (!handle_arguments(cfgmap, prevm, ini_config))
            {
                return -2;
            }

            // re-initialize runtime configuration object
            if (prevm.count("hpx:config"))
                rtcfg_.reconfigure(prevm["hpx:config"].as<std::string>());
            else
                rtcfg_.reconfigure("");

            // Make sure any aliases defined on the command line get used
            // for the option analysis below.
            std::vector<std::string> cfg;
            if (prevm.count("hpx:ini"))
            {
                cfg = prevm["hpx:ini"].as<std::vector<std::string>>();
                cfgmap.add(cfg);
            }

            // append ini options from command line
            std::copy(ini_config_.begin(), ini_config_.end(),
                std::back_inserter(cfg));

            // enable logging if invoked requested from command line
            std::vector<std::string> ini_config_logging;
            enable_logging_settings(prevm, ini_config_logging);

            std::copy(ini_config_logging.begin(), ini_config_logging.end(),
                std::back_inserter(cfg));

            rtcfg_.reconfigure(cfg);
        }

        // Re-run program option analysis, ini settings (such as aliases)
        // will be considered now.

        // Now re-parse the command line using the node number (if given).
        // This will additionally detect any --hpx:N:foo options.
        hpx::program_options::options_description help;
        std::vector<std::string> unregistered_options;

        if (!parse_commandline(rtcfg_, desc_cmdline, argv[0], args, vm_,
                error_mode | util::report_missing_config_file, &help,
                &unregistered_options))
        {
            return -1;
        }

        // break into debugger, if requested
        handle_attach_debugger();

        // handle all --hpx:foo and --hpx:*:foo options
        if (!handle_arguments(cfgmap, vm_, ini_config_))
        {
            return -2;
        }

        // store unregistered command line and arguments
        store_command_line(argc, argv);
        store_unregistered_options(argv[0], unregistered_options);

        // add all remaining ini settings to the global configuration
        rtcfg_.reconfigure(ini_config_);

        // help can be printed only after the runtime mode has been set
        if (handle_help_options(help))
        {
            return 1;    // exit application gracefully
        }

        // print version/copyright information
        if (vm_.count("hpx:version"))
        {
            if (!version_printed_)
            {
                detail::print_version(std::cout);
                version_printed_ = true;
            }

            return 1;
        }

        // print configuration information (static and dynamic)
        if (vm_.count("hpx:info"))
        {
            if (!info_printed_)
            {
                detail::print_info(std::cout, *this);
                info_printed_ = true;
            }

            return 1;
        }

        // all is good
        return 0;
    }
}}}    // namespace hpx::local::detail
