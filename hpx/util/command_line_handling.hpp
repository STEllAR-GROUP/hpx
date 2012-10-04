//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_COMMAND_LINE_HANDLING_OCT_04_2012_0800AM)
#define HPX_UTIL_COMMAND_LINE_HANDLING_OCT_04_2012_0800AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/hpx_init.hpp>

#include <boost/program_options.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    /// 
    HPX_EXPORT int command_line_handling(
        boost::program_options::options_description& desc_cmdline,
        int argc, char* argv[], std::vector<std::string> ini_config,
        hpx::runtime_mode mode, hpx_main_type& f, 
        boost::program_options::variables_map& vm,
        util::runtime_configuration& rtcfg, std::size_t& num_threads,
        std::size_t& num_localities, std::string& queuing);
}}

#endif

