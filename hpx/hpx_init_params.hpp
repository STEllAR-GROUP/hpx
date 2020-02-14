//  Copyright (c)      2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx_init_params.hpp

#ifndef HPX_HPX_INIT_PARAMS_HPP
#define HPX_HPX_INIT_PARAMS_HPP

#include <hpx/config.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/program_options.hpp>
#include <hpx/resource_partitioner/partitioner_fwd.hpp>
#include <hpx/runtime_configuration/runtime_mode.hpp>
#include <hpx/runtime/shutdown_function.hpp>
#include <hpx/runtime/startup_function.hpp>

#include <functional>
#include <string>
#include <vector>

#ifndef DOXYGEN
///////////////////////////////////////////////////////////////////////////////
// One of these functions must be implemented by the application for the
// console locality.
int hpx_main();
int hpx_main(int argc, char** argv);
int hpx_main(hpx::program_options::variables_map& vm);
#endif

///////////////////////////////////////////////////////////////////////////////
/// \cond NOINTERNAL
namespace hpx_startup
{
    // As an alternative, the user can provide a function hpx_startup::user_main,
    // which is semantically equivalent to the plain old C-main.
    int user_main();
    int user_main(int argc, char** argv);
}
/// \endcond

///////////////////////////////////////////////////////////////////////////////
/// \namespace hpx
namespace hpx
{
    /// \cond NOINTERNAL
    namespace resource {
        // Utilities to init the thread_pools of the resource partitioner
        using rp_callback_type = hpx::util::function_nonser<void(
                hpx::resource::partitioner&)>;
    }
    /// \endcond

    namespace detail
    {
        HPX_EXPORT void on_exit() noexcept;
        HPX_EXPORT void on_abort(int signal) noexcept;
        // Default params to initialize the init_params struct
        static char app_name[] = HPX_APPLICATION_STRING;
        static const hpx::program_options::options_description default_desc =
            hpx::program_options::options_description(
                "Usage: " HPX_APPLICATION_STRING " [options]");
        static startup_function_type default_startup = startup_function_type();
        static shutdown_function_type default_shutdown = shutdown_function_type();
        static int dummy_argc = 1;
        // TODO: make it only one parameter, probably add a cast
        static char *default_argv[2] = { detail::app_name , nullptr };
        static char **dummy_argv = default_argv;
    }

#ifndef DOXYGEN
    typedef int (*hpx_main_type)(hpx::program_options::variables_map&);
    typedef int (*hpx_user_main_type)(int argc, char** argv);
#endif

    /// \struct init_params
    /// \brief A struct to contain the hpx::init() parameters
    struct init_params {
        // Parameters
        std::reference_wrapper<hpx::program_options::options_description const>
            desc_cmdline = detail::default_desc;
        std::vector<std::string> cfg;
        startup_function_type& startup = detail::default_startup;
        shutdown_function_type& shutdown = detail::default_startup;
        hpx::runtime_mode mode = ::hpx::runtime_mode_default;
        hpx::resource::partitioner_mode rp_mode = ::hpx::resource::mode_default;
        hpx::resource::rp_callback_type rp_callback;
    };
}

#endif
