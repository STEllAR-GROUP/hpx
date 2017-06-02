//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_COMMAND_LINE_HANDLING_HPP
#define HPX_UTIL_COMMAND_LINE_HANDLING_HPP

#include <hpx/config.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/hpx_user_main_config.hpp>
#include <hpx/runtime/runtime_mode.hpp>
#include <hpx/util/function.hpp>
#include <hpx/util/manage_config.hpp>
#include <hpx/util/runtime_configuration.hpp>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/format.hpp>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    struct command_line_handling
    {
        command_line_handling(char const* argv0)
          : rtcfg_(argv0),
            ini_config_(hpx_startup::user_main_config(std::vector<std::string>(0))),
            hpx_main_f_(static_cast<hpx_main_type>(::hpx_main)),
            node_(std::size_t(-1)),
            num_threads_(1),
            num_cores_(1),
            num_localities_(1),
            pu_step_(1),
            pu_offset_(std::size_t(-1)),
            numa_sensitive_(0)
        {}

        command_line_handling()
          : rtcfg_(nullptr),
            node_(std::size_t(-1)),
            num_threads_(0),
            num_cores_(1),
            num_localities_(1),
            pu_step_(1),
            pu_offset_(std::size_t(-1)),
            numa_sensitive_(0)
        {}

        int call(boost::program_options::options_description const& desc_cmdline,
            int argc, char** argv);

        boost::program_options::variables_map vm_;
        util::runtime_configuration rtcfg_;

        std::vector<std::string> ini_config_;
        util::function_nonser<
            int(boost::program_options::variables_map& vm)
        > hpx_main_f_;

        std::size_t node_;
        std::size_t num_threads_;
        std::size_t num_cores_;
        std::size_t num_localities_;
        std::size_t pu_step_;
        std::size_t pu_offset_;
        std::string queuing_;
        std::string affinity_domain_;
        std::string affinity_bind_;
        std::size_t numa_sensitive_;

    protected:
        bool handle_arguments(util::manage_config& cfgmap,
            boost::program_options::variables_map& vm,
            std::vector<std::string>& ini_config, std::size_t& node,
            bool initial = false);

        void store_command_line(int argc, char** argv);
        void store_unregistered_options(std::string const& cmd_name,
            std::vector<std::string> const& unregistered_options);
        bool handle_help_options(
            boost::program_options::options_description const& help);

        void handle_attach_debugger();
    };

#if defined(HPX_HAVE_HWLOC)
    void handle_print_bind(boost::program_options::variables_map const& vm,
        std::size_t num_threads);
#endif

    void handle_list_parcelports();

    void HPX_EXPORT attach_debugger();
}}

#endif /*HPX_UTIL_COMMAND_LINE_HANDLING_HPP*/
