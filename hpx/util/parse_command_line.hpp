//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_PARSE_COMMAND_LINE_NOV_30_2011_0652PM)
#define HPX_UTIL_PARSE_COMMAND_LINE_NOV_30_2011_0652PM

#include <hpx/config.hpp>
#include <hpx/runtime/runtime_mode.hpp>
#include <hpx/util/ini.hpp>

#include <boost/program_options.hpp>

#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    enum commandline_error_mode
    {
        return_on_error,
        rethrow_on_error,
        allow_unregistered,
        report_missing_config_file = 0x80
    };

    ///////////////////////////////////////////////////////////////////////////
    // parse the command line
    HPX_API_EXPORT bool parse_commandline(
        hpx::util::section const& rtcfg,
        boost::program_options::options_description const& app_options,
        std::string const& cmdline, boost::program_options::variables_map& vm,
        std::size_t node, int error_mode = return_on_error,
        hpx::runtime_mode mode = runtime_mode_default,
        boost::program_options::options_description* visible = nullptr,
        std::vector<std::string>* unregistered_options = nullptr);

    HPX_API_EXPORT bool parse_commandline(
        hpx::util::section const& rtcfg,
        boost::program_options::options_description const& app_options,
        int argc, char** argv, boost::program_options::variables_map& vm,
        std::size_t node, int error_mode = return_on_error,
        hpx::runtime_mode mode = runtime_mode_default,
        boost::program_options::options_description* visible = nullptr,
        std::vector<std::string>* unregistered_options = nullptr);

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

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        inline std::string enquote(std::string const& arg)
        {
            if (arg.find_first_of(" \t") != std::string::npos)
                return std::string("\"") + arg + "\"";
            return arg;
        }
    }
}}

#endif
