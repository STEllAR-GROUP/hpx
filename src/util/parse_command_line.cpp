//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime.hpp>
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
        // Additional command line parser which interprets '@something' as an
        // option "options-file" with the value "something". Additionally we
        // map any option -N (where N is a integer) to --node=N.
        inline std::pair<std::string, std::string>
        option_parser(std::string const& s)
        {
            if ('@' == s[0])
                return std::make_pair(std::string("options-file"), s.substr(1));

            if ('-' == s[0] && s.size() > 1 && std::isdigit(s[1])) {
                try {
                    // test, whether next argument is an integer
                    boost::lexical_cast<std::size_t>(s.substr(1));
                    return std::make_pair(std::string("node"), s.substr(1));
                }
                catch (boost::bad_lexical_cast const&) {
                    ;   // ignore
                }
            }
            return std::pair<std::string, std::string>();
        }

        ///////////////////////////////////////////////////////////////////////
        // Read all options from a given config file, parse and add them to the
        // given variables_map
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
            if (vm.count("options-file")) {
                std::vector<std::string> const &cfg_files =
                    vm["options-file"].as<std::vector<std::string> >();
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
        boost::program_options::options_description const& app_options,
        int argc, char *argv[], boost::program_options::variables_map& vm,
        commandline_error_mode error_mode,
        hpx::runtime_mode mode,
        boost::program_options::options_description* visible,
        std::vector<std::string>* unregistered_options)
    {
        using boost::program_options::options_description;
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
                ("help,h", value<std::string>()->implicit_value("minimal"),
                    "print out program usage (default: this message), possible "
                    "values: 'full' (additionally prints options from components)")
                ("version,v", "print out HPX version and copyright information")
                ("options-file", value<std::vector<std::string> >()->composing(),
                    "specify a file containing command line options "
                    "(alternatively: @filepath)")
            ;

            options_description hpx_options(
                "HPX options (additionally allowed in an options file)");
            options_description hidden_options("Hidden options");

            switch (mode) {
            case runtime_mode_default:
                hpx_options.add_options()
                    ("worker,w", "run this instance in worker mode")
                    ("console,c", "run this instance in console mode")
                    ("connect", "run this instance in worker mode, but connecting late")
                ;
                break;

            case runtime_mode_worker:
            case runtime_mode_console:
            case runtime_mode_connect:
                // If the runtime for this application is always run in
                // worker mode, silently ignore the worker option for
                // hpx_pbs compatibility.
                hidden_options.add_options()
                    ("worker,w", "run this instance in worker mode")
                    ("console,c", "run this instance in console mode")
                    ("connect", "run this instance in worker mode, but connecting late")
                ;
                break;

            case runtime_mode_invalid:
            default:
                throw std::logic_error("Invalid runtime mode specified");
            }

            // general options definitions
            hpx_options.add_options()
                ("run-agas-server,r",
                  "run AGAS server as part of this runtime instance")
                ("run-hpx-main",
                  "run the hpx_main function, regardless of locality mode")
                ("agas,a", value<std::string>(),
                  "the IP address the AGAS server is running on, "
                  "expected format: `address:port' (default: "
                  "127.0.0.1:7910)")
                ("run-agas-server-only", "run only the AGAS server")
                ("hpx,x", value<std::string>(),
                  "the IP address the HPX parcelport is listening on, "
                  "expected format: `address:port' (default: "
                  "127.0.0.1:7910)")
                ("nodefile", value<std::string>(),
                  "the file name of a node file to use (list of nodes, one "
                  "node name per line and core)")
                ("nodes", value<std::vector<std::string> >()->multitoken(),
                  "the (space separated) list of the nodes to use (usually "
                  "this is extracted from a node file)")
                ("ifsuffix", value<std::string>(),
                  "suffix to append to host names in order to resolve them "
                  "to the proper network interconnect")
                ("ifprefix", value<std::string>(),
                  "prefix to prepend to host names in order to resolve them "
                  "to the proper network interconnect")
                ("iftransform", value<std::string>(),
                  "sed-style search and replace (s/search/replace/) used to "
                  "transform host names to the proper network interconnect")
                ("localities,l", value<std::size_t>(),
                  "the number of localities to wait for at application "
                  "startup (default: 1)")
                ("node", value<std::size_t>(),
                  "number of the node this locality is run on "
                  "(must be unique, alternatively: -1, -2, etc.)")
#if defined(HPX_HAVE_HWLOC)
                ("pu-offset", value<std::size_t>(),
                  "the first processing unit this instance of HPX should be "
                  "run on (default: 0)")
                ("pu-step", value<std::size_t>(),
                  "the step between used processing unit numbers for this "
                  "instance of HPX (default: 1)")
#endif
                ("threads,t", value<std::size_t>(),
                  "the number of operating system threads to dedicate as "
                  "shepherd threads for this HPX locality (default: 1)")
                ("queueing,q", value<std::string>(),
                  "the queue scheduling policy to use, options are 'global/g', "
                  "'local/l', 'priority_local/pr', 'abp/a', 'priority_abp', "
                  "'hierarchy/h', and 'periodic/pe' (default: priority_local/p)")
                ("hierarchy-arity", value<std::size_t>(),
                  "the arity of the of the thread queue tree, valid for "
                   "--queueing=hierarchy only (default: 2)")
                ("high-priority-threads", value<std::size_t>(),
                  "the number of operating system threads maintaining a high "
                  "priority queue (default: number of OS threads), valid for "
                  "--queueing=priority_local only")
                ("numa-sensitive",
                  "makes the priority_local scheduler NUMA sensitive, valid for "
                  "--queueing=priority_local only")
            ;

            options_description config_options("HPX configuration options");
            config_options.add_options()
                ("app-config,p", value<std::string>(),
                  "load the specified application configuration (ini) file")
                ("hpx-config", value<std::string>()->default_value(""),
                  "load the specified hpx configuration (ini) file")
                ("ini,I", value<std::vector<std::string> >()->composing(),
                  "add a configuration definition to the default runtime "
                  "configuration")
                ("exit", "exit after configuring the runtime")
            ;

            options_description debugging_options("HPX debugging options");
            debugging_options.add_options()
                ("list-symbolic-names", "list all registered symbolic "
                  "names after startup")
                ("list-component-types", "list all dynamic component types "
                  "after startup")
                ("dump-config-initial", "print the initial runtime configuration")
                ("dump-config", "print the final runtime configuration")
                ("debug-hpx-log", value<std::string>()->implicit_value("cout"),
                  "enable all messages on the HPX log channel and send all "
                  "HPX logs to the target destination")
                ("debug-agas-log", value<std::string>()->implicit_value("cout"),
                  "enable all messages on the AGAS log channel and send all "
                  "AGAS logs to the target destination")
                // enable debug output from command line handling
                ("debug-clp", "debug command line processing")
            ;

            options_description counter_options(
                "HPX options related to performance counters");
            counter_options.add_options()
                ("print-counter", value<std::vector<std::string> >()->composing(),
                  "print the specified performance counter either repeatedly or "
                  "before shutting down the system (see option --print-counter-interval)")
                ("print-counter-interval", value<std::size_t>(),
                  "print the performance counter(s) specified with --print-counter "
                  "repeatedly after the time interval (specified in milliseconds) "
                  "(default: 0, which means print once at shutdown)")
                ("list-counters", "list the names of all registered performance "
                  "counters")
                ("list-counter-infos", "list the description of all registered "
                  "performance counters")
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
                        .style(unix_style)
                        .extra_parser(detail::option_parser),
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
            if (visible && vm.count("help")) {
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
            app_options, static_cast<int>(args.size()), argv.get(), vm,
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
        hpx::util::runtime_configuration& cfg = hpx::get_runtime().get_config();
        if (cfg.has_entry("hpx.cmd_line"))
            cmdline = cfg.get_entry("hpx.cmd_line", "");

        return parse_commandline(app_options, cmdline, vm, allow_unregistered);
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
}}
