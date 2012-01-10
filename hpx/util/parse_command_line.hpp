//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_PARSE_COMMAND_LINE_NOV_30_2011_0652PM)
#define HPX_UTIL_PARSE_COMMAND_LINE_NOV_30_2011_0652PM

#include <hpx/hpx_fwd.hpp>
#include <boost/program_options.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    enum commandline_error_mode
    {
        return_on_error,
        rethrow_on_error,
        allow_unregistered
    };

    ///////////////////////////////////////////////////////////////////////////
    // parse the command line
    HPX_API_EXPORT bool parse_commandline(
        boost::program_options::options_description const& app_options,
        std::string const& cmdline, boost::program_options::variables_map& vm,
        commandline_error_mode error_mode = return_on_error,
        hpx::runtime_mode mode = runtime_mode_default,
        boost::program_options::options_description* visible = 0,
        std::vector<std::string>* unregistered_options = 0);

    ///////////////////////////////////////////////////////////////////////////
    /// \section cmdline_options_sec HPX Application Command Line Options
    ///
    /// The predefined command line options of any application using \a hpx#init
    /// are described below.
    ///
    /// <code>
    /// HPX options (allowed on command line only):
    ///   -h [ --help ]         print out program usage (this message)
    ///   -v [ --version ]      print out HPX version and copyright information
    ///   --options-file arg    specify a file containing command line options
    ///                         (alternatively: @filepath)
    ///
    /// HPX options (additionally allowed in an options file):
    ///   -w [ --worker ]             run this instance in worker mode
    ///   -c [ --console ]            run this instance in console mode
    ///   --connect                   run this instance in worker mode, but connecting
    ///                               late
    ///   -r [ --run-agas-server ]    run AGAS server as part of this runtime instance
    ///   --run-hpx-main              run the hpx_main function, regardless of locality
    ///                               mode
    ///   -a [ --agas ] arg           the IP address the AGAS server is running on,
    ///                               expected format: `address:port' (default:
    ///                               127.0.0.1:7910)
    ///   -x [ --hpx ] arg            the IP address the HPX parcelport is listening
    ///                               on, expected format: `address:port' (default:
    ///                               127.0.0.1:7910)
    ///   -l [ --localities ] arg     the number of localities to wait for at
    ///                               application startup (default: 1)
    ///   -t [ --threads ] arg        the number of operating system threads to spawn
    ///                               for this HPX locality (default: 1)
    ///   -q [ --queueing ] arg       the queue scheduling policy to use, options are
    ///                               `global/g', `local/l', `priority_local/p' and
    ///                               `abp/a' (default: priority_local/p)
    ///   --high-priority-threads arg the number of operating system threads
    ///                               maintaining a high priority queue (default:
    ///                               number of OS threads), valid for
    ///                               --queueing=priority_local only
    ///   -p [ --app-config ] arg     load the specified application configuration
    ///                               (ini) file
    ///   --hpx-config arg            load the specified hpx configuration (ini) file
    ///   -I [ --ini ] arg            add a configuration definition to the default
    ///                               runtime configuration
    ///   --dump-config               print the runtime configuration
    ///   --exit                      exit after configuring the runtime
    ///   -P [ --print-counter ] arg  print the specified performance counter before
    ///                               shutting down the system
    ///   --list-counters             list all registered performance counters
    ///   --node arg                  number of the node this locality is run on (must
    ///                               be unique, alternatively: -1, -2, etc.)
    ///</code>
    HPX_API_EXPORT bool parse_commandline(
        boost::program_options::options_description const& app_options,
        int argc, char *argv[], boost::program_options::variables_map& vm,
        commandline_error_mode error_mode = return_on_error,
        hpx::runtime_mode mode = runtime_mode_default,
        boost::program_options::options_description* visible = 0,
        std::vector<std::string>* unregistered_options = 0);

    ///////////////////////////////////////////////////////////////////////////
    // retrieve the command line arguments for the current locality
    HPX_API_EXPORT bool retrieve_commandline_arguments(
        boost::program_options::options_description const& app_options,
        boost::program_options::variables_map& vm);

    ///////////////////////////////////////////////////////////////////////////
    // retrieve the command line arguments for the current locality
    HPX_API_EXPORT bool retrieve_commandline_arguments(
        std::string const& appname, boost::program_options::variables_map& vm);
}}

#endif
