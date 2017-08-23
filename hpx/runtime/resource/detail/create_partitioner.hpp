//  Copyright (c)      2017 Shoshana Jakobovits
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_DETAIL_CREATE_PARTITIONER_AUG_10_2017_1116AM)
#define HPX_DETAIL_CREATE_PARTITIONER_AUG_10_2017_1116AM

#include <hpx/config.hpp>
#include <hpx/runtime/resource/partitioner_fwd.hpp>
#include <hpx/runtime/runtime_mode.hpp>

#include <boost/program_options.hpp>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#if !defined(HPX_EXPORTS)
// This function must be implemented by the application.
int hpx_main(boost::program_options::variables_map& vm);
typedef int (*hpx_main_type)(boost::program_options::variables_map&);
#endif

namespace hpx { namespace resource { namespace detail
{
    // if the resource partitioner is accessed before the HPX runtime has started
    // then on first access, this function should be used, to ensure that command line
    // affinity binding options are honored. Use this function signature only once
    // and thereafter use the parameter free version.
    HPX_EXPORT partitioner& create_partitioner(
        util::function_nonser<
            int(boost::program_options::variables_map& vm)
        > const& f,
        boost::program_options::options_description const& desc_cmdline,
        int argc, char** argv, std::vector<std::string> ini_config,
        resource::partitioner_mode rpmode = resource::mode_default,
        runtime_mode mode = runtime_mode_default,
        bool check = true);

#if !defined(HPX_EXPORTS)
    inline partitioner& create_partitioner(
        int argc, char** argv,
        resource::partitioner_mode rpmode = resource::mode_default,
        runtime_mode mode = runtime_mode_default, bool check = true)
    {
        boost::program_options::options_description desc_cmdline(
            std::string("Usage: ") + HPX_APPLICATION_STRING + " [options]");

        return create_partitioner(static_cast<hpx_main_type>(::hpx_main),
            desc_cmdline, argc, argv, std::vector<std::string>(),
            rpmode, mode, check);
    }

    inline partitioner &create_partitioner(
        int argc, char **argv, std::vector<std::string> ini_config,
        resource::partitioner_mode rpmode = resource::mode_default,
        runtime_mode mode = runtime_mode_default, bool check = true)
    {
        boost::program_options::options_description desc_cmdline(
            std::string("Usage: ") + HPX_APPLICATION_STRING + " [options]");

        return create_partitioner(static_cast<hpx_main_type>(::hpx_main),
            desc_cmdline, argc, argv, std::move(ini_config),
            rpmode, mode, check);
    }

    ///////////////////////////////////////////////////////////////////////////////
    inline partitioner &create_partitioner(
        boost::program_options::options_description const& desc_cmdline,
        int argc, char **argv,
        resource::partitioner_mode rpmode = resource::mode_default,
        runtime_mode mode = runtime_mode_default, bool check = true)
    {
        return create_partitioner(static_cast<hpx_main_type>(::hpx_main),
            desc_cmdline, argc, argv, std::vector<std::string>(),
            rpmode, mode, check);
    }

    inline partitioner &create_partitioner(
        boost::program_options::options_description const& desc_cmdline,
        int argc, char **argv, std::vector<std::string> ini_config,
        resource::partitioner_mode rpmode = resource::mode_default,
        runtime_mode mode = runtime_mode_default, bool check = true)
    {
        return create_partitioner(static_cast<hpx_main_type>(::hpx_main),
            desc_cmdline, argc, argv, std::move(ini_config), rpmode, mode,
            check);
    }
#endif
}}}

#endif
