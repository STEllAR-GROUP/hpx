//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_COMMAND_LINE_HANDLING_OCT_04_2012_0800AM)
#define HPX_UTIL_COMMAND_LINE_HANDLING_OCT_04_2012_0800AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/manage_config.hpp>
#include <hpx/util/runtime_configuration.hpp>

#include <boost/program_options.hpp>
#include <boost/foreach.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    struct command_line_handling
    {
        command_line_handling(hpx::runtime_mode mode,
                HPX_STD_FUNCTION<int(boost::program_options::variables_map& vm)> const& f,
                std::vector<std::string> const& ini_config,
                char const* argv0)
          : rtcfg_(argv0),
            mode_(mode),
            ini_config_(ini_config),
            hpx_main_f_(f),
            num_threads_(1),
            num_cores_(1),
            num_localities_(1)
        {
            BOOST_FOREACH(std::string const& e, ini_config)
                rtcfg_.parse("<user supplied config>", e, true, false);
        }

        int call(boost::program_options::options_description  const& desc_cmdline,
            int argc, char** argv);

        boost::program_options::variables_map vm_;
        util::runtime_configuration rtcfg_;

        hpx::runtime_mode mode_;
        std::vector<std::string> ini_config_;
        HPX_STD_FUNCTION<int(boost::program_options::variables_map& vm)> hpx_main_f_;

        std::size_t num_threads_;
        std::size_t num_cores_;
        std::size_t num_localities_;
        std::string queuing_;

    protected:
        bool handle_arguments(util::manage_config& cfgmap,
            boost::program_options::variables_map& vm,
            std::vector<std::string>& ini_config, std::size_t& node);

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
}}

#endif

