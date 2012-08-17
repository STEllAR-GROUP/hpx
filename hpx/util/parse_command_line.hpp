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
        hpx::util::section const& rtcfg,
        boost::program_options::options_description const& app_options,
        std::string const& cmdline, boost::program_options::variables_map& vm,
        commandline_error_mode error_mode = return_on_error,
        hpx::runtime_mode mode = runtime_mode_default,
        boost::program_options::options_description* visible = 0,
        std::vector<std::string>* unregistered_options = 0);

    HPX_API_EXPORT bool parse_commandline(
        hpx::util::section const& rtcfg,
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

    ///////////////////////////////////////////////////////////////////////////
    HPX_API_EXPORT std::string reconstruct_command_line(
        boost::program_options::variables_map const &vm);
}}

#endif
