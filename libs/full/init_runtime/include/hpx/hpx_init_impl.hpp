//  Copyright (c)      2018 Mikael Simberg
//  Copyright (c) 2007-2022 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/assert.hpp>
#include <hpx/functional/bind_back.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/hpx_main_winsocket.hpp>
#include <hpx/hpx_user_main_config.hpp>
#include <hpx/init_runtime/detail/run_or_start.hpp>
#include <hpx/init_runtime_local/init_runtime_local.hpp>
#include <hpx/modules/program_options.hpp>
#include <hpx/prefix/find_prefix.hpp>
#include <hpx/runtime_configuration/runtime_mode.hpp>
#include <hpx/runtime_local/runtime_local_fwd.hpp>
#include <hpx/runtime_local/shutdown_function.hpp>
#include <hpx/runtime_local/startup_function.hpp>

#include <csignal>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if defined(__FreeBSD__)
extern HPX_EXPORT char** freebsd_environ;
extern char** environ;
#endif

#if defined(HPX_WINDOWS) && defined(HPX_HAVE_APEX)
namespace apex {

    // force linking of the application with APEX
    HPX_SYMBOL_IMPORT std::string& version();
}    // namespace apex
#endif

namespace hpx {
    namespace detail {
        inline int init_impl(
            hpx::function<int(hpx::program_options::variables_map&)> const& f,
            int argc, char** argv, init_params const& params)
        {
            if (argc == 0 || argv == nullptr)
            {
                argc = hpx::local::detail::dummy_argc;
                argv = hpx::local::detail::dummy_argv;
            }

#if defined(HPX_WINDOWS)
            detail::init_winsocket();
#if defined(HPX_HAVE_APEX)
            // artificially force the apex shared library to be loaded by the
            // application
            apex::version();
#endif
#endif
            util::set_hpx_prefix(HPX_PREFIX);
#if defined(__FreeBSD__)
            freebsd_environ = environ;
#endif
            // set a handler for std::abort, std::at_quick_exit, and std::atexit
            std::signal(SIGABRT, detail::on_abort);
            std::atexit(detail::on_exit);
#if defined(HPX_HAVE_CXX11_STD_QUICK_EXIT)
            std::at_quick_exit(detail::on_exit);
#endif
            return detail::run_or_start(f, argc, argv, params, true);
        }
    }    // namespace detail

    /// \brief Main entry point for launching the HPX runtime system.
    ///
    /// This is the main entry point for any HPX application. This function
    /// (or one of its overloads below) should be called from the users `main()`
    /// function. It will set up the HPX runtime environment and schedule the
    /// function given by \p f as a HPX thread.
    inline int init(std::function<int(hpx::program_options::variables_map&)> f,
        int argc, char** argv, init_params const& params)
    {
        return detail::init_impl(HPX_MOVE(f), argc, argv, params);
    }

    /// \brief Main entry point for launching the HPX runtime system.
    ///
    /// This is the main entry point for any HPX application. This function
    /// (or one of its overloads below) should be called from the users `main()`
    /// function. It will set up the HPX runtime environment and schedule the
    /// function given by \p f as a HPX thread.
    inline int init(std::function<int(int, char**)> f, int argc, char** argv,
        init_params const& params)
    {
        std::function<int(hpx::program_options::variables_map&)> main_f =
            hpx::bind_back(detail::init_helper, HPX_MOVE(f));
        return detail::init_impl(HPX_MOVE(main_f), argc, argv, params);
    }

    /// \brief Main entry point for launching the HPX runtime system.
    ///
    /// This is the main entry point for any HPX application. This function
    /// (or one of its overloads below) should be called from the users `main()`
    /// function. It will set up the HPX runtime environment and schedule the
    /// function given by \p f as a HPX thread.
    inline int init(int argc, char** argv, init_params const& params)
    {
        std::function<int(hpx::program_options::variables_map&)> main_f =
            static_cast<hpx_main_type>(::hpx_main);
        return detail::init_impl(HPX_MOVE(main_f), argc, argv, params);
    }

    /// \brief Main entry point for launching the HPX runtime system.
    ///
    /// This is the main entry point for any HPX application. This function
    /// (or one of its overloads below) should be called from the users `main()`
    /// function. It will set up the HPX runtime environment and schedule the
    /// function given by \p f as a HPX thread.
    inline int init(
        std::nullptr_t, int argc, char** argv, init_params const& params)
    {
        hpx::function<int(hpx::program_options::variables_map&)> main_f;
        return detail::init_impl(HPX_MOVE(main_f), argc, argv, params);
    }

    /// \brief Main entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main entry point, which can be used to set up the
    /// runtime for an HPX application (the runtime system will be set up in
    /// console mode or worker mode depending on the command line settings).
    inline int init(init_params const& params)
    {
        hpx::function<int(hpx::program_options::variables_map&)> main_f =
            static_cast<hpx_main_type>(::hpx_main);
        return detail::init_impl(HPX_MOVE(main_f),
            hpx::local::detail::dummy_argc, hpx::local::detail::dummy_argv,
            params);
    }
}    // namespace hpx
