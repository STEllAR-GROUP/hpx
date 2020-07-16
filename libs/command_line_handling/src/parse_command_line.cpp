//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/command_line_handling/parse_command_line.hpp>
#include <hpx/datastructures/any.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/filesystem.hpp>
#include <hpx/runtime_configuration/ini.hpp>
#include <hpx/util/from_string.hpp>

#include <cctype>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util {
    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        inline std::string trim_whitespace(std::string const& s)
        {
            using size_type = std::string::size_type;

            size_type first = s.find_first_not_of(" \t");
            if (std::string::npos == first)
                return std::string();

            size_type last = s.find_last_not_of(" \t");
            return s.substr(first, last - first + 1);
        }

        ///////////////////////////////////////////////////////////////////////
        // All command line options which are normally formatted as --hpx:foo
        // should be usable as --hpx:N:foo, where N is the node number this
        // option should be exclusively used for.
        bool handle_node_specific_option(std::string const& s, std::size_t node,
            std::pair<std::string, std::string>& opt)
        {
            // any option not starting with --hpx: will be handled elsewhere
            constexpr char const hpx_prefix[] = "--hpx:";
            constexpr std::string::size_type const hpx_prefix_len =
                sizeof(hpx_prefix) - 1;

            if (s.size() < hpx_prefix_len ||
                s.compare(0, hpx_prefix_len, hpx_prefix) != 0 ||
                !std::isdigit(s[hpx_prefix_len]))    // -V557
            {
                return false;
            }

            // any --hpx: option without a second ':' is handled elsewhere as well
            std::string::size_type p = s.find_first_of(':', hpx_prefix_len);
            if (p == std::string::npos)
                return false;

            if (hpx::util::from_string<std::size_t>(
                    s.substr(hpx_prefix_len, p - hpx_prefix_len),
                    std::size_t(-1)) == node)
            {
                // this option is for the current locality only
                std::string::size_type p1 = s.find_first_of('=', p);
                if (p1 != std::string::npos)
                {
                    // the option has a value
                    std::string o(
                        "hpx:" + trim_whitespace(s.substr(p + 1, p1 - p - 1)));
                    std::string v(trim_whitespace(s.substr(p1 + 1)));
                    opt = std::make_pair(o, v);
                }
                else
                {
                    // no value
                    std::string o("hpx:" + trim_whitespace(s.substr(p + 1)));
                    opt = std::make_pair(o, std::string());
                }
                return true;
            }

            // This option is specifically not for us, so we return an option
            // which will be silently ignored.
            opt = std::make_pair(std::string("hpx:ignore"), std::string());
            return true;
        }

        ///////////////////////////////////////////////////////////////////////
        // Handle aliasing of command line options based on information stored
        // in the ini-configuration
        std::pair<std::string, std::string> handle_aliasing(
            util::section const& ini, std::string const& option)
        {
            std::pair<std::string, std::string> result;

            std::string opt(trim_whitespace(option));
            if (opt.size() < 2 || opt[0] != '-')
                return result;

            util::section const* sec =
                ini.get_section("hpx.commandline.aliases");
            if (nullptr == sec)
                return result;    // no alias mappings are defined

            // we found shortcut option definitions, try to find mapping
            std::string expand_to;
            std::string::size_type start_at = 2;
            bool long_option = false;
            if (opt.size() > 2 && opt[1] != '-')
            {
                // short option with value: first two letters have to match
                expand_to = trim_whitespace(
                    sec->get_entry(opt.substr(0, start_at), ""));
            }
            else
            {
                // short option (no value) or long option
                if (opt[1] == '-')
                {
                    start_at = opt.find_last_of('=');
                    long_option = true;
                }

                if (start_at != std::string::npos)
                {
                    expand_to = trim_whitespace(
                        sec->get_entry(opt.substr(0, start_at), ""));
                }
                else
                {
                    expand_to = trim_whitespace(sec->get_entry(opt, ""));
                }
            }

            if (expand_to.size() < 2 || expand_to.substr(0, 2) != "--")
                return result;    // no sensible alias is defined for this option
            expand_to.erase(0, 2);

            std::string::size_type p = expand_to.find_first_of('=');
            if (p != std::string::npos)
            {
                // the option alias defines its own value
                std::string o(trim_whitespace(expand_to.substr(0, p)));
                std::string v(trim_whitespace(expand_to.substr(p + 1)));
                result = std::make_pair(o, v);
            }
            else if (start_at != std::string::npos && start_at < opt.size())
            {
                // extract value from original option
                result = std::make_pair(
                    expand_to, opt.substr(start_at + (long_option ? 1 : 0)));
            }
            else
            {
                // no value
                result = std::make_pair(expand_to, std::string());
            }

            return result;
        }

        ///////////////////////////////////////////////////////////////////////
        // Additional command line parser which interprets '@something' as an
        // option "options-file" with the value "something". Additionally we
        // resolve defined command line option aliases.
        struct option_parser
        {
            option_parser(util::section const& ini, std::size_t node)
              : ini_(ini)
              , node_(node)
            {
            }

            std::pair<std::string, std::string> operator()(
                std::string const& s) const
            {
                // handle special syntax for configuration files @filename
                if ('@' == s[0])
                    return std::make_pair(
                        std::string("hpx:options-file"), s.substr(1));

                // handle node specific options
                std::pair<std::string, std::string> opt;
                if (handle_node_specific_option(s, node_, opt))
                    return opt;

                // handle aliasing, if enabled
                if (ini_.get_entry("hpx.commandline.aliasing", "1") == "1")
                    return handle_aliasing(ini_, s);

                return opt;
            }

            util::section const& ini_;
            std::size_t node_;
        };

        ///////////////////////////////////////////////////////////////////////
        hpx::program_options::basic_command_line_parser<char>&
        get_commandline_parser(
            hpx::program_options::basic_command_line_parser<char>& p, int mode)
        {
            if ((mode & ~util::report_missing_config_file) ==
                util::allow_unregistered)
                return p.allow_unregistered();
            return p;
        }

        ///////////////////////////////////////////////////////////////////////
        // Read all options from a given config file, parse and add them to the
        // given variables_map
        bool read_config_file_options(std::string const& filename,
            hpx::program_options::options_description const& desc,
            hpx::program_options::variables_map& vm, util::section const& rtcfg,
            std::size_t node, int error_mode)
        {
            std::ifstream ifs(filename.c_str());
            if (!ifs.is_open())
            {
                if (error_mode & util::report_missing_config_file)
                {
                    std::cerr
                        << "hpx::init: command line warning: command line "
                           "options file not found ("
                        << filename << ")" << std::endl;
                }
                return false;
            }

            std::vector<std::string> options;
            std::string line;
            while (std::getline(ifs, line))
            {
                // skip empty lines
                std::string::size_type pos = line.find_first_not_of(" \t");
                if (pos == std::string::npos)
                    continue;

                // strip leading and trailing whitespace
                line = trim_whitespace(line);

                // skip comment lines
                if ('#' != line[0])
                {
                    std::string::size_type p1 = line.find_first_of(" \t");
                    if (p1 != std::string::npos)
                    {
                        // rebuild the line connecting the parts with a '='
                        line = trim_whitespace(line.substr(0, p1)) + '=' +
                            trim_whitespace(line.substr(p1));
                    }
                    options.push_back(line);
                }
            }

            // add options to parsed settings
            if (!options.empty())
            {
                using hpx::program_options::basic_command_line_parser;
                using hpx::program_options::command_line_parser;
                using hpx::program_options::store;
                using hpx::program_options::value;
                using namespace hpx::program_options::command_line_style;

                store(detail::get_commandline_parser(
                          command_line_parser(options)
                              .options(desc)
                              .style(unix_style)
                              .extra_parser(detail::option_parser(rtcfg, node)),
                          error_mode)
                          .run(),
                    vm);
                notify(vm);
            }
            return true;
        }

        // try to find a config file somewhere up the filesystem hierarchy
        // starting with the input file path. This allows to use a general
        // <app_name>.cfg file for all executables in a certain project.
        void handle_generic_config_options(std::string appname,
            hpx::program_options::variables_map& vm,
            hpx::program_options::options_description const& desc_cfgfile,
            util::section const& ini, std::size_t node, int error_mode)
        {
            if (appname.empty())
                return;

            filesystem::path dir(filesystem::initial_path());
            filesystem::path app(appname);
            appname = filesystem::basename(app.filename());

            // walk up the hierarchy, trying to find a file <appname>.cfg
            while (!dir.empty())
            {
                filesystem::path filename = dir / (appname + ".cfg");
                bool result = read_config_file_options(filename.string(),
                    desc_cfgfile, vm, ini, node,
                    error_mode & ~util::report_missing_config_file);
                if (result)
                    break;    // break on the first options file found

                    // Boost filesystem and C++17 filesystem behave differently
                    // here. Boost filesystem returns an empty path for
                    // "/".parent_path() whereas C++17 filesystem will keep
                    // returning "/".
#if !defined(HPX_FILESYSTEM_HAVE_BOOST_FILESYSTEM_COMPATIBILITY)
                auto dir_prev = dir;
                dir = dir.parent_path();    // chop off last directory part
                if (dir_prev == dir)
                    break;
#else
                dir = dir.parent_path();    // chop off last directory part
#endif
            }
        }

        // handle all --options-config found on the command line
        void handle_config_options(hpx::program_options::variables_map& vm,
            hpx::program_options::options_description const& desc_cfgfile,
            util::section const& ini, std::size_t node, int error_mode)
        {
            using hpx::program_options::options_description;
            if (vm.count("hpx:options-file"))
            {
                std::vector<std::string> const& cfg_files =
                    vm["hpx:options-file"].as<std::vector<std::string>>();

                for (std::string const& cfg_file : cfg_files)
                {
                    // parse a single config file and store the results
                    read_config_file_options(
                        cfg_file, desc_cfgfile, vm, ini, node, error_mode);
                }
            }
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    // parse the command line
    bool parse_commandline(util::section const& rtcfg,
        hpx::program_options::options_description const& app_options,
        std::string const& arg0, std::vector<std::string> const& args,
        hpx::program_options::variables_map& vm, std::size_t node,
        int error_mode, hpx::runtime_mode mode,
        hpx::program_options::options_description* visible,
        std::vector<std::string>* unregistered_options)
    {
        using hpx::program_options::basic_command_line_parser;
        using hpx::program_options::command_line_parser;
        using hpx::program_options::options_description;
        using hpx::program_options::parsed_options;
        using hpx::program_options::positional_options_description;
        using hpx::program_options::store;
        using hpx::program_options::value;
        using namespace hpx::program_options::command_line_style;

        try
        {
            // clang-format off
            options_description cmdline_options(
                "HPX options (allowed on command line only)");
            cmdline_options.add_options()
                ("hpx:help", value<std::string>()->implicit_value("minimal"),
                    "print out program usage (default: this message), possible "
                    "values: 'full' (additionally prints options from components)")
                ("hpx:version", "print out HPX version and copyright information")
                ("hpx:info", "print out HPX configuration information")
                ("hpx:options-file", value<std::vector<std::string> >()->composing(),
                    "specify a file containing command line options "
                    "(alternatively: @filepath)")
            ;

            options_description hpx_options(
                "HPX options (additionally allowed in an options file)");
            options_description hidden_options("Hidden options");
            // clang-format on

            switch (mode)
            {
            case runtime_mode::default_:
#if defined(HPX_HAVE_NETWORKING)
                // clang-format off
                hpx_options.add_options()
                    ("hpx:worker", "run this instance in worker mode")
                    ("hpx:console", "run this instance in console mode")
                    ("hpx:connect", "run this instance in worker mode, "
                         "but connecting late")
                ;
#else
                hpx_options.add_options()
                    ("hpx:console", "run this instance in console mode")
                ;
                // clang-format on
#endif
                break;

#if defined(HPX_HAVE_NETWORKING)
            case runtime_mode::worker:
            case runtime_mode::console:
            case runtime_mode::connect:
                // If the runtime for this application is always run in
                // worker mode, silently ignore the worker option for
                // hpx_pbs compatibility.
                // clang-format off
                hidden_options.add_options()
                    ("hpx:worker", "run this instance in worker mode")
                    ("hpx:console", "run this instance in console mode")
                    ("hpx:connect", "run this instance in worker mode, "
                        "but connecting late")
                ;
                // clang-format on
                break;
#else
            case runtime_mode::console:
                // clang-format off
                hidden_options.add_options()
                    ("hpx:console", "run this instance in console mode")
                ;
                // clang-format on
                break;
#endif
            case runtime_mode::local:
                break;

            case runtime_mode::invalid:
            default:
                throw hpx::detail::command_line_error(
                    "Invalid runtime mode specified");
            }

            // Always add the option to start the local runtime
            hpx_options.add_options()("hpx:local",
                "run this instance in local mode (experimental; certain "
                "functionality not available at runt-time)");

            // general options definitions
            // clang-format off
            hpx_options.add_options()
                ("hpx:run-hpx-main",
                  "run the hpx_main function, regardless of locality mode")
#if defined(HPX_HAVE_NETWORKING)
                ("hpx:agas", value<std::string>(),
                  "the IP address the AGAS root server is running on, "
                  "expected format: `address:port' (default: "
                  "127.0.0.1:7910)")
                ("hpx:hpx", value<std::string>(),
                  "the IP address the HPX parcelport is listening on, "
                  "expected format: `address:port' (default: "
                  "127.0.0.1:7910)")
                ("hpx:nodefile", value<std::string>(),
                  "the file name of a node file to use (list of nodes, one "
                  "node name per line and core)")
                ("hpx:nodes", value<std::vector<std::string> >()->multitoken(),
                  "the (space separated) list of the nodes to use (usually "
                  "this is extracted from a node file)")
                ("hpx:endnodes", "this can be used to end the list of nodes "
                  "specified using the option --hpx:nodes")
                ("hpx:ifsuffix", value<std::string>(),
                  "suffix to append to host names in order to resolve them "
                  "to the proper network interconnect")
                ("hpx:ifprefix", value<std::string>(),
                  "prefix to prepend to host names in order to resolve them "
                  "to the proper network interconnect")
                ("hpx:iftransform", value<std::string>(),
                  "sed-style search and replace (s/search/replace/) used to "
                  "transform host names to the proper network interconnect")
#endif
#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
                ("hpx:localities", value<std::size_t>(),
                  "the number of localities to wait for at application "
                  "startup (default: 1)")
#endif
#if defined(HPX_HAVE_NETWORKING)
                ("hpx:node", value<std::size_t>(),
                  "number of the node this locality is run on "
                  "(must be unique, alternatively: -0, -1, ..., -9)")
                ("hpx:ignore-batch-env", "ignore batch environment variables "
                 "(implied by --hpx:use-process-mask)")
                ("hpx:expect-connecting-localities",
                  "this locality expects other localities to dynamically connect "
                  "(default: false if the number of localities is equal to one, "
                  "true if the number of initial localities is larger than 1)")
#endif
                ("hpx:pu-offset", value<std::size_t>(),
                  "the first processing unit this instance of HPX should be "
                  "run on (default: 0), valid for "
                  "--hpx:queuing=local, --hpx:queuing=abp-priority, "
                  "--hpx:queuing=static, --hpx:queuing=static-priority, "
                  "and --hpx:queuing=local-priority only")
                ("hpx:pu-step", value<std::size_t>(),
                  "the step between used processing unit numbers for this "
                  "instance of HPX (default: 1), valid for "
                  "--hpx:queuing=local, --hpx:queuing=abp-priority, "
                  "--hpx:queuing=static, --hpx:queuing=static-priority "
                  "and --hpx:queuing=local-priority only")
                ("hpx:affinity", value<std::string>(),
                  "the affinity domain the OS threads will be confined to, "
                  "possible values: pu, core, numa, machine (default: pu), valid for "
                  "--hpx:queuing=local, --hpx:queuing=abp-priority, "
                  "--hpx:queuing=static, --hpx:queuing=static-priority "
                  " and --hpx:queuing=local-priority only")
                ("hpx:bind", value<std::vector<std::string> >()->composing(),
                  "the detailed affinity description for the OS threads, see "
                  "the documentation for a detailed description of possible "
                  "values. Do not use with --hpx:pu-step, --hpx:pu-offset, or "
                  "--hpx:affinity options. Implies --hpx:numa-sensitive=1"
                  "(--hpx:bind=none disables defining thread affinities).")
                ("hpx:use-process-mask", "use the process mask to restrict"
                 "available hardware resources (implies "
                 "--hpx:ignore-batch-environment)")
                ("hpx:print-bind",
                  "print to the console the bit masks calculated from the "
                  "arguments specified to all --hpx:bind options.")
                ("hpx:threads", value<std::string>(),
                 "the number of operating system threads to spawn for this HPX "
                 "locality (default: 1, using 'all' will spawn one thread for "
                 "each processing unit")
                ("hpx:cores", value<std::string>(),
                 "the number of cores to utilize for this HPX "
                 "locality (default: 'all', i.e. the number of cores is based on "
                 "the number of total cores in the system)")
                ("hpx:queuing", value<std::string>(),
                  "the queue scheduling policy to use, options are "
                  "'local', 'local-priority-fifo','local-priority-lifo', "
                  "'abp-priority-fifo', 'abp-priority-lifo', 'static', and "
                  "'static-priority' (default: 'local-priority'; "
                  "all option values can be abbreviated)")
                ("hpx:high-priority-threads", value<std::size_t>(),
                  "the number of operating system threads maintaining a high "
                  "priority queue (default: number of OS threads), valid for "
                  "--hpx:queuing=local-priority,--hpx:queuing=static-priority, "
                  " and --hpx:queuing=abp-priority only)")
                ("hpx:numa-sensitive", value<std::size_t>()->implicit_value(0),
                  "makes the local-priority scheduler NUMA sensitive ("
                  "allowed values: 0 - no NUMA sensitivity, 1 - allow only for "
                  "boundary cores to steal across NUMA domains, 2 - "
                  "no cross boundary stealing is allowed (default value: 0)")
            ;

            options_description config_options("HPX configuration options");
            config_options.add_options()
                ("hpx:app-config", value<std::string>(),
                  "load the specified application configuration (ini) file")
                ("hpx:config", value<std::string>()->default_value(""),
                  "load the specified hpx configuration (ini) file")
                ("hpx:ini", value<std::vector<std::string> >()->composing(),
                  "add a configuration definition to the default runtime "
                  "configuration")
                ("hpx:exit", "exit after configuring the runtime")
            ;

            options_description debugging_options("HPX debugging options");
            debugging_options.add_options()
                ("hpx:dump-config-initial", "print the initial runtime configuration")
                ("hpx:dump-config", "print the final runtime configuration")
                // enable debug output from command line handling
                ("hpx:debug-clp", "debug command line processing")
#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
                ("hpx:list-symbolic-names", "list all registered symbolic "
                  "names after startup")
                ("hpx:list-component-types", "list all dynamic component types "
                  "after startup")
                ("hpx:debug-hpx-log", value<std::string>()->implicit_value("cout"),
                  "enable all messages on the HPX log channel and send all "
                  "HPX logs to the target destination")
                ("hpx:debug-agas-log", value<std::string>()->implicit_value("cout"),
                  "enable all messages on the AGAS log channel and send all "
                  "AGAS logs to the target destination")
                ("hpx:debug-parcel-log", value<std::string>()->implicit_value("cout"),
                  "enable all messages on the parcel transport log channel and send all "
                  "parcel transport logs to the target destination")
                ("hpx:debug-timing-log", value<std::string>()->implicit_value("cout"),
                  "enable all messages on the timing log channel and send all "
                  "timing logs to the target destination")
                ("hpx:debug-app-log", value<std::string>()->implicit_value("cout"),
                  "enable all messages on the application log channel and send all "
                  "application logs to the target destination")
#endif
#if defined(_POSIX_VERSION) || defined(HPX_WINDOWS)
                ("hpx:attach-debugger",
                  value<std::string>()->implicit_value("startup"),
                  "wait for a debugger to be attached, possible values: "
                  "off, startup, exception or test-failure (default: startup)")
#endif
#if defined(HPX_HAVE_NETWORKING)
                ("hpx:list-parcel-ports", "list all available parcel-ports")
#endif
            ;

#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
            options_description counter_options(
                "HPX options related to performance counters");
            counter_options.add_options()
                ("hpx:print-counter", value<std::vector<std::string> >()->composing(),
                  "print the specified performance counter either repeatedly "
                  "and/or at the times specified by --hpx:print-counter-at "
                    "(see also option --hpx:print-counter-interval)")
                ("hpx:print-counter-reset",
                        value<std::vector<std::string> >()->composing(),
                  "print the specified performance counter either repeatedly "
                  "and/or at the times specified by --hpx:print-counter-at, "
                    "reset the counter after the "
                    "value is queried (see also option --hpx:print-counter-interval)")
                ("hpx:print-counter-interval", value<std::size_t>(),
                  "print the performance counter(s) specified with --hpx:print-counter "
                  "repeatedly after the time interval (specified in milliseconds) "
                  "(default: 0, which means print once at shutdown)")
                ("hpx:print-counter-destination", value<std::string>(),
                  "print the performance counter(s) specified with --hpx:print-counter "
                  "to the given file (default: console (cout), "
                  "possible values: 'cout' (console), 'none' (no output), or "
                  "any file name")
                ("hpx:list-counters", value<std::string>()->implicit_value("minimal"),
                  "list the names of all registered performance counters, "
                  "possible values:\n"
                  "   'minimal' (prints counter name skeletons)\n"
                  "   'full' (prints all available counter names)")
                ("hpx:list-counter-infos",
                    value<std::string>()->implicit_value("minimal"),
                  "list the description of all registered performance counters, "
                  "possible values:\n"
                  "   'minimal' (prints infos for counter name skeletons)\n"
                  "   'full' (prints all available counter infos)")
                ("hpx:print-counter-format", value<std::string>(),
                  "print the performance counter(s) specified with --hpx:print-counter "
                  "in a given format (default: normal)")
                ("hpx:csv-header",
                  "print the performance counter(s) specified with --hpx:print-counter "
                  "with header when format specified with --hpx:print-counter-format"
                  "is csv or csv-short")
                ("hpx:no-csv-header",
                  "print the performance counter(s) specified with --hpx:print-counter "
                  "without header when format specified with --hpx:print-counter-format"
                  "is csv or csv-short")
                ("hpx:print-counter-at",
                    value<std::vector<std::string> >()->composing(),
                  "print the performance counter(s) specified with "
                  "--hpx:print-counter (or --hpx:print-counter-reset) at the given "
                  "point in time, possible "
                  "argument values: 'startup', 'shutdown' (default), 'noshutdown'")
                ("hpx:reset-counters",
                  "reset all performance counter(s) specified with --hpx:print-counter "
                  "after they have been evaluated")
                ("hpx:print-counters-locally",
                  "each locality prints only its own local counters")
                ("hpx:print-counter-types",
                  "append counter type description to generated output")
            ;
#endif

            hidden_options.add_options()
                ("hpx:ignore", "this option will be silently ignored")
            ;
            // clang-format off

            // construct the overall options description and parse the
            // command line
            options_description desc_cmdline;
            options_description positional_options;
            desc_cmdline
                .add(app_options).add(cmdline_options)
                .add(hpx_options)
                .add(config_options).add(debugging_options)
                .add(hidden_options)
            ;
#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
            desc_cmdline.add(counter_options);
#endif

            options_description desc_cfgfile;
            desc_cfgfile
                .add(app_options).add(hpx_options)
                .add(config_options)
                .add(debugging_options).add(hidden_options)
            ;
#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
            desc_cfgfile.add(counter_options);
#endif

            if (rtcfg.get_entry("hpx.commandline.allow_unknown", "0") == "0")
            {
                // move all positional options into the hpx:positional option
                // group
                positional_options_description pd;
                pd.add("hpx:positional", -1);

                positional_options.add_options()
                    ("hpx:positional", value<std::vector<std::string> >(),
                      "positional options")
                ;
                desc_cmdline.add(positional_options);
                desc_cfgfile.add(positional_options);

                // parse command line, allow for unregistered options this point
                parsed_options opts(detail::get_commandline_parser(
                        command_line_parser(args)
                            .options(desc_cmdline)
                            .positional(pd)
                            .style(unix_style)
                            .extra_parser(detail::option_parser(rtcfg, node)),
                        error_mode
                    ).run()
                );

                // collect unregistered options, if needed
                if (unregistered_options) {
                    using hpx::program_options::collect_unrecognized;
                    using hpx::program_options::exclude_positional;
                    *unregistered_options =
                        collect_unrecognized(opts.options, exclude_positional);
                }

                store(opts, vm);
            }
            else
            {
                // parse command line, allow for unregistered options this point
                parsed_options opts(detail::get_commandline_parser(
                        command_line_parser(args)
                            .options(desc_cmdline)
                            .style(unix_style)
                            .extra_parser(detail::option_parser(rtcfg, node)),
                        error_mode
                    ).run()
                );

                // collect unregistered options, if needed
                if (unregistered_options) {
                    using hpx::program_options::collect_unrecognized;
                    using hpx::program_options::include_positional;
                    *unregistered_options =
                        collect_unrecognized(opts.options, include_positional);
                }

                store(opts, vm);
            }

            if (vm.count("hpx:help"))
            {
                // collect help information
                if (visible) {
                    (*visible)
                        .add(app_options).add(cmdline_options)
                        .add(hpx_options)
                        .add(debugging_options).add(config_options)
                    ;
#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
                    (*visible).add(counter_options);
#endif
                }
                return true;
            }

            notify(vm);

            detail::handle_generic_config_options(
                arg0, vm, desc_cfgfile, rtcfg, node, error_mode);
            detail::handle_config_options(
                vm, desc_cfgfile, rtcfg, node, error_mode);
        }
        catch (std::exception const& e) {
            if (error_mode & rethrow_on_error)
                throw;

            std::cerr << "hpx::init: exception caught: "
                      << e.what() << std::endl;
            return false;
        }
        return true;
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        std::string extract_arg0(std::string const& cmdline)
        {
            std::string::size_type p = cmdline.find_first_of(" \t");
            if (p != std::string::npos)
            {
                return cmdline.substr(0, p);
            }
            return cmdline;
        }
    }

    bool parse_commandline(
        util::section const& rtcfg,
        hpx::program_options::options_description const& app_options,
        std::string const& cmdline, hpx::program_options::variables_map& vm,
        std::size_t node, int error_mode, hpx::runtime_mode mode,
        hpx::program_options::options_description* visible,
        std::vector<std::string>* unregistered_options)
    {
        using namespace hpx::program_options;
#if defined(HPX_WINDOWS)
        std::vector<std::string> args = split_winmain(cmdline);
#else
        std::vector<std::string> args = split_unix(cmdline);
#endif
        return parse_commandline(rtcfg, app_options,
            detail::extract_arg0(cmdline), args, vm, node, error_mode, mode,
            visible, unregistered_options);
    }

    ///////////////////////////////////////////////////////////////////////////
    std::string embed_in_quotes(std::string const& s)
    {
        char quote = (s.find_first_of('"') != std::string::npos) ? '\'' : '"';

        if (s.find_first_of("\t ") != std::string::npos)
            return quote + s + quote;
        return s;
    }

    void add_as_option(std::string& command_line, std::string const& k,
        std::string const& v)
    {
        command_line += "--" + k;
        if (!v.empty())
            command_line += "=" + v;
    }

    std::string
    reconstruct_command_line(hpx::program_options::variables_map const &vm)
    {
        std::string command_line;
        for (auto const& v : vm)
        {
            hpx::program_options::any const& value = v.second.value();
            if (hpx::program_options::any_cast<std::string>(&value)) {
                add_as_option(command_line, v.first,
                    embed_in_quotes(v.second.as<std::string>()));
                if (!command_line.empty())
                    command_line += " ";
            }
            else if (hpx::program_options::any_cast<double>(&value)) {
                add_as_option(command_line, v.first,
                    std::to_string(v.second.as<double>()));
                if (!command_line.empty())
                    command_line += " ";
            }
            else if (hpx::program_options::any_cast<int>(&value)) {
                add_as_option(command_line, v.first,
                    std::to_string(v.second.as<int>()));
                if (!command_line.empty())
                    command_line += " ";
            }
            else if (hpx::program_options::any_cast<std::vector<std::string>>(
                         &value))
            {
                auto const& vec = v.second.as<std::vector<std::string>>();
                for (std::string const& e : vec)
                {
                    add_as_option(command_line, v.first, embed_in_quotes(e));
                    if (!command_line.empty())
                        command_line += " ";
                }
            }
        }
        return command_line;
    }
}}
