//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime.hpp>
#include <hpx/util/ini.hpp>
#include <hpx/util/parse_command_line.hpp>
#include <hpx/util/runtime_configuration.hpp>

#include <string>
#include <stdexcept>
#include <cctype>
#include <fstream>
#include <vector>

#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/scoped_array.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        inline std::string
        trim_whitespace (std::string const &s)
        {
            typedef std::string::size_type size_type;

            size_type first = s.find_first_not_of(" \t");
            if (std::string::npos == first)
                return std::string();

            size_type last = s.find_last_not_of(" \t");
            return s.substr(first, last - first + 1);
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

            util::section const* sec = ini.get_section("hpx.commandline");
            if (NULL == sec)
                return result;     // no alias mappings are defined

            // we found a shortcut option, try to find mapping
            std::string expand_to;
            std::string::size_type start_at = 2;
            bool long_option = false;
            if (opt.size() > 2 && opt[1] != '-') {
                // short option with value: first two letters have to match
                expand_to = trim_whitespace(sec->get_entry(opt.substr(0, start_at), ""));
            }
            else {
                // short option (no value) or long option
                if (opt[1] == '-') {
                    start_at = opt.find_last_of("=");
                    long_option = true;
                }

                if (start_at != std::string::npos) {
                    expand_to = trim_whitespace(
                        sec->get_entry(opt.substr(0, start_at), ""));
                }
                else {
                    expand_to = trim_whitespace(sec->get_entry(opt, ""));
                }
            }

            if (expand_to.size() < 2 || expand_to.substr(0, 2) != "--")
                return result;     // no sensible alias is defined for this option
            expand_to.erase(0, 2);

            std::string::size_type p = expand_to.find_first_of('=');
            if (p != std::string::npos) {
                // the option alias defines its own value
                std::string o(trim_whitespace(expand_to.substr(0, p)));
                std::string v(trim_whitespace(expand_to.substr(p+1)));
                result = std::make_pair(o, v);
            }
            else if (start_at != std::string::npos && start_at < opt.size()) {
                // extract value from original option
                result = std::make_pair(expand_to,
                    opt.substr(start_at + (long_option ? 1 : 0)));
            }
            else {
                // no value
                result = std::make_pair(expand_to, "");
            }

            return result;
        }

        ///////////////////////////////////////////////////////////////////////
        // Additional command line parser which interprets '@something' as an
        // option "options-file" with the value "something". Additionally we
        // resolve defined command line option aliases.
        struct option_parser
        {
            option_parser(util::section const& ini)
              : ini_(ini)
            {}

            std::pair<std::string, std::string> operator()(std::string const& s) const
            {
                if ('@' == s[0])
                    return std::make_pair(std::string("hpx:options-file"), s.substr(1));

                return handle_aliasing(ini_, s);
            }

            util::section const& ini_;
        };

        ///////////////////////////////////////////////////////////////////////
        // Read all options from a given config file, parse and add them to the
        // given variables_map
        bool read_config_file_options(std::string const &filename,
            boost::program_options::options_description const &desc,
            boost::program_options::variables_map &vm, bool may_fail = false)
        {
            std::ifstream ifs(filename.c_str());
            if (!ifs.is_open()) {
                if (!may_fail) {
                    std::cerr << "hpx::init: command line warning: command line "
                          "options file not found (" << filename << ")"
                        << std::endl;
                }
                return false;
            }

            std::vector<std::string> options;
            std::string line;
            while (std::getline(ifs, line)) {
                // skip empty lines
                std::string::size_type pos = line.find_first_not_of(" \t");
                if (pos == std::string::npos)
                    continue;

                // skip comment lines
                if ('#' != line[pos]) {
                    // strip leading and trailing whitespace
                    line = trim_whitespace(line);

                    std::string::size_type p1 = line.find_first_of(" \t");
                    if (p1 != std::string::npos) {
                        // rebuild the line connecting the parts with a '='
                        line = trim_whitespace(line.substr(0, p1)) + '=' +
                            trim_whitespace(line.substr(p1));
                    }
                    options.push_back(line);
                }
            }

            // add options to parsed settings
            if (options.size() > 0) {
                using boost::program_options::value;
                using boost::program_options::store;
                using boost::program_options::command_line_parser;
                using namespace boost::program_options::command_line_style;

                store(command_line_parser(options)
                    .options(desc).style(unix_style).run(), vm);
                notify(vm);
            }
            return true;
        }

        // try to find a config file somewhere up the filesystem hierarchy
        // starting with the input file path. This allows to use a general wave.cfg
        // file for all files in a certain project.
        void handle_generic_config_options(std::string appname,
            boost::program_options::variables_map& vm,
            boost::program_options::options_description const& desc_cfgfile)
        {
            if (appname.empty())
                return;

            boost::filesystem::path dir (boost::filesystem::initial_path());
            boost::filesystem::path app (appname);
            appname = boost::filesystem::basename(app.filename());

            // walk up the hierarchy, trying to find a file appname.cfg
            while (!dir.empty()) {
                boost::filesystem::path filename = dir / (appname + ".cfg");
                if (read_config_file_options(filename.string(), desc_cfgfile, vm, true))
                    break;    // break on the first options file found

                dir = dir.parent_path();    // chop off last directory part
            }
        }

        // handle all --options-config found on the command line
        void handle_config_options(boost::program_options::variables_map& vm,
            boost::program_options::options_description const& desc_cfgfile)
        {
            using boost::program_options::options_description;
            if (vm.count("hpx:options-file")) {
                std::vector<std::string> const &cfg_files =
                    vm["hpx:options-file"].as<std::vector<std::string> >();
                BOOST_FOREACH(std::string const& cfg_file, cfg_files)
                {
                    // parse a single config file and store the results
                    read_config_file_options(cfg_file, desc_cfgfile, vm);
                }
            }
        }

        ///////////////////////////////////////////////////////////////////////
        boost::program_options::basic_command_line_parser<char>&
        get_commandline_parser(
            boost::program_options::basic_command_line_parser<char>& p,
            commandline_error_mode mode)
        {
            return (mode == allow_unregistered) ? p.allow_unregistered() : p;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // parse the command line
    bool parse_commandline(
        util::section const& rtcfg,
        boost::program_options::options_description const& app_options,
        int argc, char *argv[], boost::program_options::variables_map& vm,
        commandline_error_mode error_mode,
        hpx::runtime_mode mode,
        boost::program_options::options_description* visible,
        std::vector<std::string>* unregistered_options)
    {
        using boost::program_options::options_description;
        using boost::program_options::positional_options_description;
        using boost::program_options::value;
        using boost::program_options::store;
        using boost::program_options::command_line_parser;
        using boost::program_options::parsed_options;
        using boost::program_options::basic_command_line_parser;
        using namespace boost::program_options::command_line_style;

        try {
            options_description cmdline_options(
                "HPX options (allowed on command line only)");
            cmdline_options.add_options()
                ("hpx:help", value<std::string>()->implicit_value("minimal"),
                    "print out program usage (default: this message), possible "
                    "values: 'full' (additionally prints options from components)")
                ("hpx:version", "print out HPX version and copyright information")
                ("hpx:options-file", value<std::vector<std::string> >()->composing(),
                    "specify a file containing command line options "
                    "(alternatively: @filepath)")
            ;

            options_description hpx_options(
                "HPX options (additionally allowed in an options file)");
            options_description hidden_options("Hidden options");

            switch (mode) {
            case runtime_mode_default:
                hpx_options.add_options()
                    ("hpx:worker", "run this instance in worker mode")
                    ("hpx:console", "run this instance in console mode")
                    ("hpx:connect", "run this instance in worker mode, but connecting late")
                ;
                break;

            case runtime_mode_worker:
            case runtime_mode_console:
            case runtime_mode_connect:
                // If the runtime for this application is always run in
                // worker mode, silently ignore the worker option for
                // hpx_pbs compatibility.
                hidden_options.add_options()
                    ("hpx:worker", "run this instance in worker mode")
                    ("hpx:console", "run this instance in console mode")
                    ("hpx:connect", "run this instance in worker mode, but connecting late")
                ;
                break;

            case runtime_mode_invalid:
            default:
                throw std::logic_error("Invalid runtime mode specified");
            }

            // general options definitions
            hpx_options.add_options()
                ("hpx:run-agas-server",
                  "run AGAS server as part of this runtime instance")
                ("hpx:run-hpx-main",
                  "run the hpx_main function, regardless of locality mode")
                ("hpx:agas", value<std::string>(),
                  "the IP address the AGAS root server is running on, "
                  "expected format: `address:port' (default: "
                  "127.0.0.1:7910)")
                ("hpx:run-agas-server-only", "run only the AGAS server")
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
                ("hpx:ifsuffix", value<std::string>(),
                  "suffix to append to host names in order to resolve them "
                  "to the proper network interconnect")
                ("hpx:ifprefix", value<std::string>(),
                  "prefix to prepend to host names in order to resolve them "
                  "to the proper network interconnect")
                ("hpx:iftransform", value<std::string>(),
                  "sed-style search and replace (s/search/replace/) used to "
                  "transform host names to the proper network interconnect")
                ("hpx:localities", value<std::size_t>(),
                  "the number of localities to wait for at application "
                  "startup (default: 1)")
                ("hpx:node", value<std::size_t>(),
                  "number of the node this locality is run on "
                  "(must be unique, alternatively: -0, -1, ..., -9)")
#if defined(HPX_HAVE_HWLOC) || defined(BOOST_WINDOWS)
                ("hpx:pu-offset", value<std::size_t>(),
                  "the first processing unit this instance of HPX should be "
                  "run on (default: 0)")
                ("hpx:pu-step", value<std::size_t>(),
                  "the step between used processing unit numbers for this "
                  "instance of HPX (default: 1)")
#endif
                ("hpx:threads", value<std::string>(),
                 "the number of operating system threads to spawn for this HPX "
                 "locality (default: 1, using 'all' will spawn one thread for "
                 "each processing unit")
                ("hpx:queuing", value<std::string>(),
                  "the queue scheduling policy to use, options are 'global/g', "
                  "'local/l', 'priority_local/pr', 'abp/a', 'priority_abp', "
                  "'hierarchy/h', and 'periodic/pe' (default: priority_local/p)")
                ("hpx:hierarchy-arity", value<std::size_t>(),
                  "the arity of the of the thread queue tree, valid for "
                   "--hpx:queuing=hierarchy only (default: 2)")
                ("hpx:high-priority-threads", value<std::size_t>(),
                  "the number of operating system threads maintaining a high "
                  "priority queue (default: number of OS threads), valid for "
                  "--hpx:queuing=priority_local only")
                ("hpx:numa-sensitive",
                  "makes the priority_local scheduler NUMA sensitive, valid for "
                  "--hpx:queuing=priority_local only")
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
                ("hpx:list-symbolic-names", "list all registered symbolic "
                  "names after startup")
                ("hpx:list-component-types", "list all dynamic component types "
                  "after startup")
                ("hpx:dump-config-initial", "print the initial runtime configuration")
                ("hpx:dump-config", "print the final runtime configuration")
                ("hpx:debug-hpx-log", value<std::string>()->implicit_value("cout"),
                  "enable all messages on the HPX log channel and send all "
                  "HPX logs to the target destination")
                ("hpx:debug-agas-log", value<std::string>()->implicit_value("cout"),
                  "enable all messages on the AGAS log channel and send all "
                  "AGAS logs to the target destination")
                // enable debug output from command line handling
                ("hpx:debug-clp", "debug command line processing")
            ;

            options_description counter_options(
                "HPX options related to performance counters");
            counter_options.add_options()
                ("hpx:print-counter", value<std::vector<std::string> >()->composing(),
                  "print the specified performance counter either repeatedly or "
                  "before shutting down the system (see option --hpx:print-counter-interval)")
                ("hpx:print-counter-interval", value<std::size_t>(),
                  "print the performance counter(s) specified with --hpx:print-counter "
                  "repeatedly after the time interval (specified in milliseconds) "
                  "(default: 0, which means print once at shutdown)")
                ("hpx:print-counter-destination", value<std::string>(),
                  "print the performance counter(s) specified with --hpx:print-counter "
                  "to the given file (default: console)")
                ("hpx:list-counters", "list the names of all registered performance "
                  "counters")
                ("hpx:list-counter-infos", "list the description of all registered "
                  "performance counters")
            ;

            // move all positional options into the hpx:positional option group
            positional_options_description pd;
            pd.add("hpx:positional", -1);

            hidden_options.add_options()
                ("hpx:positional", value<std::vector<std::string> >(),
                  "positional options")
            ;

            // construct the overall options description and parse the
            // command line
            options_description desc_cmdline;
            desc_cmdline
                .add(app_options).add(cmdline_options)
                .add(hpx_options).add(counter_options)
                .add(config_options).add(debugging_options)
                .add(hidden_options)
            ;

            // parse command line, allow for unregistered options this point
            parsed_options opts(
                detail::get_commandline_parser(
                    command_line_parser(argc, argv)
                        .options(desc_cmdline)
                        .positional(pd)
                        .style(unix_style)
                        .extra_parser(detail::option_parser(rtcfg)),
                    error_mode
                ).run()
            );

            // collect unregistered options, if needed
            if (unregistered_options) {
                using boost::program_options::collect_unrecognized;
                using boost::program_options::exclude_positional;
                *unregistered_options =
                    collect_unrecognized(opts.options, exclude_positional);
            }

            store(opts, vm);
            notify(vm);

            options_description desc_cfgfile;
            desc_cfgfile
                .add(app_options).add(hpx_options)
                .add(counter_options).add(config_options)
                .add(debugging_options).add(hidden_options);

            detail::handle_generic_config_options(argv[0], vm, desc_cfgfile);
            detail::handle_config_options(vm, desc_cfgfile);

            // print help screen
            if (visible && vm.count("hpx:help")) {
                (*visible)
                    .add(app_options).add(cmdline_options)
                    .add(hpx_options).add(counter_options)
                    .add(debugging_options).add(config_options)
                ;
            }
        }
        catch (std::exception const& e) {
            if (error_mode == rethrow_on_error)
                throw;

            std::cerr << "hpx::init: exception caught: "
                      << e.what() << std::endl;
            return false;
        }
        return true;
    }

    ///////////////////////////////////////////////////////////////////////////
    bool parse_commandline(
        util::section const& rtcfg,
        boost::program_options::options_description const& app_options,
        std::string const& cmdline, boost::program_options::variables_map& vm,
        commandline_error_mode error_mode, hpx::runtime_mode mode,
        boost::program_options::options_description* visible,
        std::vector<std::string>* unregistered_options)
    {
        using namespace boost::program_options;
#if defined(BOOST_WINDOWS)
        std::vector<std::string> args = split_winmain(cmdline);
#else
        std::vector<std::string> args = split_unix(cmdline);
#endif

        boost::scoped_array<char*> argv(new char* [args.size()]);
        for (std::size_t i = 0; i < args.size(); ++i)
            argv[i] = const_cast<char*>(args[i].c_str());

        return parse_commandline(
            rtcfg, app_options, static_cast<int>(args.size()), argv.get(), vm,
            error_mode, mode, visible, unregistered_options);
    }

    ///////////////////////////////////////////////////////////////////////////
    // retrieve the command line arguments for the current locality
    bool retrieve_commandline_arguments(
        boost::program_options::options_description const& app_options,
        boost::program_options::variables_map& vm)
    {
        // To make this example at least minimally useful we analyze the
        // command line options the application instance has been started with
        // on this locality.
        //
        // The command line for this application instance is available from
        // this configuration section:
        //
        //     [hpx]
        //     cmd_line=....
        //
        std::string cmdline;
        hpx::util::section& cfg = hpx::get_runtime().get_config();
        if (cfg.has_entry("hpx.cmd_line"))
            cmdline = cfg.get_entry("hpx.cmd_line", "");

        return parse_commandline(cfg, app_options, cmdline, vm, allow_unregistered);
    }

    ///////////////////////////////////////////////////////////////////////////
    // retrieve the command line arguments for the current locality
    bool retrieve_commandline_arguments(
        std::string const& appname, boost::program_options::variables_map& vm)
    {
        using boost::program_options::options_description;

        options_description desc_commandline(
            "Usage: " + appname +  " [options]");

        return retrieve_commandline_arguments(desc_commandline, vm);
    }

    ///////////////////////////////////////////////////////////////////////////
    std::string embed_in_quotes(std::string const& s)
    {
        std::string result;
        char quote = (s.find_first_of("\"") != std::string::npos) ? '\'' : '\"';

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
    reconstruct_command_line(boost::program_options::variables_map const &vm)
    {
        typedef std::pair<std::string, boost::program_options::variable_value>
            value_type;

        std::string command_line;
        BOOST_FOREACH(value_type const& v, vm)
        {
            boost::any const& value = v.second.value();
            if (boost::any_cast<std::string>(&value)) {
                add_as_option(command_line, v.first,
                    embed_in_quotes(v.second.as<std::string>()));
                if (!command_line.empty())
                    command_line += " ";
            }
            else if (boost::any_cast<double>(&value)) {
                add_as_option(command_line, v.first,
                    boost::lexical_cast<std::string>(v.second.as<double>()));
                if (!command_line.empty())
                    command_line += " ";
            }
            else if (boost::any_cast<int>(&value)) {
                add_as_option(command_line, v.first,
                    boost::lexical_cast<std::string>(v.second.as<int>()));
                if (!command_line.empty())
                    command_line += " ";
            }
            else if (boost::any_cast<std::vector<std::string> >(&value)) {
                std::vector<std::string> const& vec =
                    v.second.as<std::vector<std::string> >();
                BOOST_FOREACH(std::string const& e, vec)
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
