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
    namespace detail
    {
        HPX_EXPORT void on_exit() noexcept;
        HPX_EXPORT void on_abort(int signal) noexcept;
    }

#ifndef DOXYGEN
    typedef int (*hpx_main_type)(hpx::program_options::variables_map&);
    typedef int (*hpx_user_main_type)(int argc, char** argv);
#endif

    /// \struct init_params
    /// \brief A struct to contain the hpx::init() parameters
    struct init_params {
        // Default args
        char *dummy_argv[2] = { const_cast<char*>(HPX_APPLICATION_STRING), nullptr };
        using options_description = hpx::program_options::options_description;
        options_description default_desc = options_description("Usage: " HPX_APPLICATION_STRING " [options]");
        // Parameters
        util::function_nonser<int(hpx::program_options::variables_map& vm)> f;
        std::shared_ptr<options_description const>
            desc_cmdline_ptr = std::make_shared<options_description>(default_desc);
        int argc = 1;
        char** argv = dummy_argv;
        std::vector<std::string> cfg;
        startup_function_type startup = startup_function_type();
        shutdown_function_type shutdown = shutdown_function_type();
        hpx::runtime_mode mode = ::hpx::runtime_mode_default;
    };
}

#endif
