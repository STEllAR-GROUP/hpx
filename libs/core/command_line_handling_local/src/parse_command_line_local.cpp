//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/command_line_handling_local/parse_command_line_local.hpp>
#include <hpx/datastructures/any.hpp>
#include <hpx/ini/ini.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/filesystem.hpp>
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
namespace hpx { namespace local { namespace detail {
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
    // Handle aliasing of command line options based on information stored
    // in the ini-configuration
    std::pair<std::string, std::string> handle_aliasing(
        util::section const& ini, std::string const& option)
    {
        std::pair<std::string, std::string> result;

        std::string opt(trim_whitespace(option));
        if (opt.size() < 2 || opt[0] != '-')
            return result;

        util::section const* sec = ini.get_section("hpx.commandline.aliases");
        if (nullptr == sec)
            return result;    // no alias mappings are defined

        // we found shortcut option definitions, try to find mapping
        std::string expand_to;
        std::string::size_type start_at = 2;
        bool long_option = false;
        if (opt.size() > 2 && opt[1] != '-')
        {
            // short option with value: first two letters have to match
            expand_to =
                trim_whitespace(sec->get_entry(opt.substr(0, start_at), ""));
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
        option_parser(util::section const& ini, bool ignore_aliases)
          : ini_(ini)
          , ignore_aliases_(ignore_aliases)
        {
        }

        std::pair<std::string, std::string> operator()(
            std::string const& s) const
        {
            // handle special syntax for configuration files @filename
            if ('@' == s[0])
                return std::make_pair(
                    std::string("hpx:options-file"), s.substr(1));

            // handle aliasing, if enabled
            if (ini_.get_entry("hpx.commandline.aliasing", "0") == "0" ||
                ignore_aliases_)
            {
                return std::make_pair(std::string(), std::string());
            }
            return handle_aliasing(ini_, s);
        }

        util::section const& ini_;
        bool ignore_aliases_;
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
        int error_mode)
    {
        std::ifstream ifs(filename.c_str());
        if (!ifs.is_open())
        {
            if (error_mode & util::report_missing_config_file)
            {
                std::cerr << "hpx::init: command line warning: command line "
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
                          .extra_parser(detail::option_parser(
                              rtcfg, error_mode & util::ignore_aliases)),
                      error_mode & ~util::ignore_aliases)
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
        util::section const& ini, int error_mode)
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
            bool result =
                read_config_file_options(filename.string(), desc_cfgfile, vm,
                    ini, error_mode & ~util::report_missing_config_file);
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
        util::section const& ini, int error_mode)
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
                    cfg_file, desc_cfgfile, vm, ini, error_mode);
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // parse the command line
    bool parse_commandline(util::section const& rtcfg,
        hpx::program_options::options_description const& app_options,
        std::string const& arg0, std::vector<std::string> const& args,
        hpx::program_options::variables_map& vm, int error_mode,
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

            // general options definitions
            // clang-format off
            hpx_options.add_options()
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
                ("hpx:use-process-mask", "use the process mask to restrict "
                 "available hardware resources (implies "
                 "--hpx:ignore-batch-env)")
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
                ("hpx:debug-hpx-log", value<std::string>()->implicit_value("cout"),
                  "enable all messages on the HPX log channel and send all "
                  "HPX logs to the target destination")
                ("hpx:debug-timing-log", value<std::string>()->implicit_value("cout"),
                  "enable all messages on the timing log channel and send all "
                  "timing logs to the target destination")
                ("hpx:debug-app-log", value<std::string>()->implicit_value("cout"),
                  "enable all messages on the application log channel and send all "
                  "application logs to the target destination")
#if defined(_POSIX_VERSION) || defined(HPX_WINDOWS)
                ("hpx:attach-debugger",
                  value<std::string>()->implicit_value("startup"),
                  "wait for a debugger to be attached, possible values: "
                  "off, startup, exception or test-failure (default: startup)")
#endif
            ;

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

            options_description desc_cfgfile;
            desc_cfgfile
                .add(app_options).add(hpx_options)
                .add(config_options)
                .add(debugging_options).add(hidden_options)
            ;

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
                            .extra_parser(detail::option_parser(rtcfg,
                                error_mode & util::ignore_aliases)),
                         error_mode & ~util::ignore_aliases
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
                            .extra_parser(detail::option_parser(rtcfg,
                                error_mode & util::ignore_aliases)),
                        error_mode & ~util::ignore_aliases
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
                }
                return true;
            }

            notify(vm);

            detail::handle_generic_config_options(
                arg0, vm, desc_cfgfile, rtcfg, error_mode);
            detail::handle_config_options(
                vm, desc_cfgfile, rtcfg, error_mode);
        }
        catch (std::exception const& e) {
            if (error_mode & util::rethrow_on_error)
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
        int error_mode,
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
            detail::extract_arg0(cmdline), args, vm, error_mode,
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
}}}
