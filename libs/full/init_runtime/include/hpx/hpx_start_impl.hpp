//  Copyright (c)      2018 Mikael Simberg
//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/assert.hpp>
#include <hpx/functional/bind_back.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/hpx_main_winsocket.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/hpx_user_main_config.hpp>
#include <hpx/init_runtime/detail/run_or_start.hpp>
#include <hpx/modules/program_options.hpp>
#include <hpx/prefix/find_prefix.hpp>
#include <hpx/runtime_configuration/runtime_mode.hpp>
#include <hpx/runtime_fwd.hpp>
#include <hpx/runtime_local/shutdown_function.hpp>
#include <hpx/runtime_local/startup_function.hpp>

#include <csignal>
#include <cstddef>
#include <cstdlib>
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
    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is the main, non-blocking entry point for any HPX application.
    /// This function (or one of its overloads below) should be called from the
    /// users `main()` function. It will set up the HPX runtime environment and
    /// schedule the function given by \p f as an HPX thread. It will return
    /// immediately after that. Use `hpx::wait` and `hpx::stop` to synchronize
    /// with the runtime system's execution.
    inline bool start(
        util::function_nonser<int(hpx::program_options::variables_map&)> const&
            f,
        int argc, char** argv, init_params const& params)
    {
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
        return 0 == detail::run_or_start(f, argc, argv, params, false);
    }

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is the main, non-blocking entry point for any HPX application.
    /// This function (or one of its overloads below) should be called from the
    /// users `main()` function. It will set up the HPX runtime environment and
    /// schedule the function given by \p f as an HPX thread. It will return
    /// immediately after that. Use `hpx::wait` and `hpx::stop` to synchronize
    /// with the runtime system's execution.
    inline bool start(util::function_nonser<int(int, char**)> const& f,
        int argc, char** argv, init_params const& params)
    {
        util::function_nonser<int(hpx::program_options::variables_map&)>
            main_f = util::bind_back(detail::init_helper, f);
        if (argc == 0 || argv == nullptr)
        {
            return start(
                main_f, detail::dummy_argc, detail::dummy_argv, params);
        }
        return start(main_f, argc, argv, params);
    }

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is the main, non-blocking entry point for any HPX application.
    /// This function (or one of its overloads below) should be called from the
    /// users `main()` function. It will set up the HPX runtime environment and
    /// schedule the function given by \p f as an HPX thread. It will return
    /// immediately after that. Use `hpx::wait` and `hpx::stop` to synchronize
    /// with the runtime system's execution.
    inline bool start(int argc, char** argv, init_params const& params)
    {
        util::function_nonser<int(hpx::program_options::variables_map&)>
            main_f = static_cast<hpx_main_type>(::hpx_main);
        if (argc == 0 || argv == nullptr)
        {
            return start(
                main_f, detail::dummy_argc, detail::dummy_argv, params);
        }
        return start(main_f, argc, argv, params);
    }

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is the main, non-blocking entry point for any HPX application.
    /// This function (or one of its overloads below) should be called from the
    /// users `main()` function. It will set up the HPX runtime environment and
    /// schedule the function given by \p f as an HPX thread. It will return
    /// immediately after that. Use `hpx::wait` and `hpx::stop` to synchronize
    /// with the runtime system's execution.
    inline bool start(
        std::nullptr_t, int argc, char** argv, init_params const& params)
    {
        util::function_nonser<int(hpx::program_options::variables_map&)> main_f;
        if (argc == 0 || argv == nullptr)
        {
            return start(
                main_f, detail::dummy_argc, detail::dummy_argv, params);
        }
        return start(main_f, argc, argv, params);
    }

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is the main, non-blocking entry point for any HPX application.
    /// This function (or one of its overloads below) should be called from the
    /// users `main()` function. It will set up the HPX runtime environment and
    /// schedule the function given by \p f as an HPX thread. It will return
    /// immediately after that. Use `hpx::wait` and `hpx::stop` to synchronize
    /// with the runtime system's execution.
    inline bool start(init_params const& params)
    {
        util::function_nonser<int(hpx::program_options::variables_map&)>
            main_f = static_cast<hpx_main_type>(::hpx_main);
        return start(main_f, detail::dummy_argc, detail::dummy_argv, params);
    }

#if defined(HPX_HAVE_INIT_START_OVERLOADS_COMPATIBILITY)
    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is the main, non-blocking entry point for any HPX application.
    /// This function (or one of its overloads below) should be called from the
    /// users `main()` function. It will set up the HPX runtime environment and
    /// schedule the function given by \p f as an HPX thread. It will return
    /// immediately after that. Use `hpx::wait` and `hpx::stop` to synchronize
    /// with the runtime system's execution.
    inline bool start(util::function_nonser<int(
                          hpx::program_options::variables_map& vm)> const& f,
        hpx::program_options::options_description const& desc_cmdline, int argc,
        char** argv, std::vector<std::string> const& cfg,
        startup_function_type startup, shutdown_function_type shutdown,
        hpx::runtime_mode mode)
    {
        using hpx::program_options::options_description;
        hpx::init_params iparams;
        iparams.desc_cmdline = desc_cmdline;
        iparams.cfg = cfg;
        iparams.startup = std::move(startup);
        iparams.shutdown = std::move(shutdown);
        iparams.mode = mode;
        return start(f, argc, argv, iparams);
    }

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is the main, non-blocking entry point for any HPX application.
    /// This function (or one of its overloads below) should be called from the
    /// users `main()` function. It will set up the HPX runtime environment and
    /// schedule the function given by \p f as an HPX thread. It will return
    /// immediately after that. Use `hpx::wait` and `hpx::stop` to synchronize
    /// with the runtime system's execution.
    inline bool start(int (*f)(hpx::program_options::variables_map& vm),
        hpx::program_options::options_description const& desc_cmdline, int argc,
        char** argv, startup_function_type startup,
        shutdown_function_type shutdown, hpx::runtime_mode mode)
    {
        std::vector<std::string> cfg;
        using hpx::program_options::options_description;

        hpx::init_params iparams;
        iparams.desc_cmdline = desc_cmdline;
        iparams.cfg = cfg;
        iparams.startup = std::move(startup);
        iparams.shutdown = std::move(shutdown);
        iparams.mode = mode;
        return start(f, argc, argv, iparams);
    }

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main, non-blocking entry point, which can be used
    /// to set up the runtime for an HPX application (the runtime system will be
    /// set up in console mode or worker mode depending on the command line
    /// settings). It will return immediately after that. Use `hpx::wait` and
    /// `hpx::stop` to synchronize with the runtime system's execution.
    inline bool start(
        hpx::program_options::options_description const& desc_cmdline, int argc,
        char** argv, startup_function_type startup,
        shutdown_function_type shutdown, hpx::runtime_mode mode)
    {
        using hpx::program_options::options_description;

        hpx::init_params iparams;
        iparams.desc_cmdline = desc_cmdline;
        iparams.startup = std::move(startup);
        iparams.shutdown = std::move(shutdown);
        iparams.mode = mode;
        return start(argc, argv, iparams);
    }

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main, non-blocking entry point, which can be used
    /// to set up the runtime for an HPX application (the runtime system will
    /// be set up in console mode or worker mode depending on the command line
    /// settings). It will return immediately after that. Use `hpx::wait` and
    /// `hpx::stop` to synchronize with the runtime system's execution.
    inline bool start(
        hpx::program_options::options_description const& desc_cmdline, int argc,
        char** argv, std::vector<std::string> const& cfg,
        startup_function_type startup, shutdown_function_type shutdown,
        hpx::runtime_mode mode)
    {
        using hpx::program_options::options_description;

        hpx::init_params iparams;
        iparams.desc_cmdline = desc_cmdline;
        iparams.cfg = cfg;
        iparams.startup = std::move(startup);
        iparams.shutdown = std::move(shutdown);
        iparams.mode = mode;
        return start(argc, argv, iparams);
    }

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main, non-blocking entry point, which can be used
    /// to set up the runtime for an HPX application (the runtime system will
    /// be set up in console mode or worker mode depending on the command line
    /// settings). It will return immediately after that. Use `hpx::wait` and
    /// `hpx::stop` to synchronize with the runtime system's execution.
    inline bool start(int argc, char** argv,
        std::vector<std::string> const& cfg, hpx::runtime_mode mode)
    {
        hpx::init_params iparams;
        iparams.cfg = cfg;
        iparams.mode = mode;
        return start(argc, argv, iparams);
    }

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main, non-blocking entry point, which can be used
    /// to set up the runtime for an HPX application (the runtime system will
    /// be set up in console mode or worker mode depending on the command line
    /// settings). It will return immediately after that. Use `hpx::wait` and
    /// `hpx::stop` to synchronize with the runtime system's execution.
    inline bool start(
        hpx::program_options::options_description const& desc_cmdline, int argc,
        char** argv, hpx::runtime_mode mode)
    {
        using hpx::program_options::options_description;

        hpx::init_params iparams;
        iparams.desc_cmdline = desc_cmdline;
        iparams.mode = mode;
        return start(argc, argv, iparams);
    }

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main, non-blocking entry point, which can be used
    /// to set up the runtime for an HPX application (the runtime system will
    /// be set up in console mode or worker mode depending on the command line
    /// settings). It will return immediately after that. Use `hpx::wait` and
    /// `hpx::stop` to synchronize with the runtime system's execution.
    inline bool start(
        hpx::program_options::options_description const& desc_cmdline, int argc,
        char** argv, std::vector<std::string> const& cfg,
        hpx::runtime_mode mode)
    {
        using hpx::program_options::options_description;

        hpx::init_params iparams;
        iparams.desc_cmdline = desc_cmdline;
        iparams.cfg = cfg;
        iparams.mode = mode;
        return start(argc, argv, iparams);
    }

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main, non-blocking entry point, which can be used
    /// to set up the runtime for an HPX application (the runtime system will
    /// be set up in console mode or worker mode depending on the command line
    /// settings). It will return immediately after that. Use `hpx::wait` and
    /// `hpx::stop` to synchronize with the runtime system's execution.
    inline bool start(std::string const& app_name, int argc, char** argv,
        hpx::runtime_mode mode)
    {
        using hpx::program_options::options_description;
        options_description desc =
            options_description("Usage: " + app_name + " [options]");
        hpx::init_params iparams;
        iparams.desc_cmdline = desc;
        iparams.mode = mode;
        return start(argc, argv, iparams);
    }

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main, non-blocking entry point, which can be used
    /// to set up the runtime for an HPX application (the runtime system will
    /// be set up in console mode or worker mode depending on the command line
    /// settings). It will return immediately after that. Use `hpx::wait` and
    /// `hpx::stop` to synchronize with the runtime system's execution.
    inline bool start(int argc, char** argv, hpx::runtime_mode mode)
    {
        hpx::init_params iparams;
        iparams.mode = mode;
        return start(argc, argv, iparams);
    }

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main, non-blocking entry point, which can be used
    /// to set up the runtime for an HPX application (the runtime system will
    /// be set up in console mode or worker mode depending on the command line
    /// settings). It will return immediately after that. Use `hpx::wait` and
    /// `hpx::stop` to synchronize with the runtime system's execution.
    inline bool start(
        std::vector<std::string> const& cfg, hpx::runtime_mode mode)
    {
        hpx::init_params iparams;
        iparams.cfg = cfg;
        iparams.mode = mode;
        return start(detail::dummy_argc, detail::dummy_argv, iparams);
    }

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main, non-blocking entry point, which can be used
    /// to set up the runtime for an HPX application (the runtime system will
    /// be set up in console mode or worker mode depending on the command line
    /// settings). It will return immediately after that. Use `hpx::wait` and
    /// `hpx::stop` to synchronize with the runtime system's execution.
    inline bool start(int (*f)(hpx::program_options::variables_map& vm),
        std::string const& app_name, int argc, char** argv,
        hpx::runtime_mode mode)
    {
        using hpx::program_options::options_description;

        options_description desc_commandline(
            "Usage: " + app_name + " [options]");

        hpx::init_params iparams;
        iparams.desc_cmdline = desc_commandline;
        iparams.mode = mode;

        if (argc == 0 || argv == nullptr)
        {
            return start(detail::dummy_argc, detail::dummy_argv, iparams);
        }

        return start(f, argc, argv, iparams);
    }

    // Main non-blocking entry point for launching the HPX runtime system.
    inline bool start(util::function_nonser<int(int, char**)> const& f,
        std::string const& app_name, int argc, char** argv,
        hpx::runtime_mode mode)
    {
        using hpx::program_options::options_description;
        options_description desc_commandline(
            "Usage: " + app_name + " [options]");

        HPX_ASSERT(argc != 0 && argv != nullptr);

        hpx::init_params iparams;
        iparams.desc_cmdline = desc_commandline;
        iparams.mode = mode;
        return start(f, argc, argv, iparams);
    }

    inline bool start(util::function_nonser<int(int, char**)> const& f,
        int argc, char** argv, std::vector<std::string> const& cfg,
        hpx::runtime_mode mode)
    {
        HPX_ASSERT(argc != 0 && argv != nullptr);

        hpx::init_params iparams;
        iparams.cfg = cfg;
        iparams.mode = mode;
        return start(f, argc, argv, iparams);
    }

    inline bool start(util::function_nonser<int(int, char**)> const& f,
        std::vector<std::string> const& cfg, hpx::runtime_mode mode)
    {
        hpx::init_params iparams;
        iparams.cfg = cfg;
        iparams.mode = mode;
        return start(f, detail::dummy_argc, detail::dummy_argv, iparams);
    }

    inline bool start(std::nullptr_t f, std::string const& app_name, int argc,
        char** argv, hpx::runtime_mode mode)
    {
        using hpx::program_options::options_description;
        options_description desc_commandline(
            "Usage: " + app_name + " [options]");

        HPX_ASSERT(argc != 0 && argv != nullptr);

        hpx::init_params iparams;
        iparams.desc_cmdline = desc_commandline;
        iparams.mode = mode;
        return start(f, argc, argv, iparams);
    }

    inline bool start(std::nullptr_t f, int argc, char** argv,
        std::vector<std::string> const& cfg, hpx::runtime_mode mode)
    {
        HPX_ASSERT(argc != 0 && argv != nullptr);

        hpx::init_params iparams;
        iparams.cfg = cfg;
        iparams.mode = mode;
        return start(f, argc, argv, iparams);
    }

    inline bool start(std::nullptr_t f, std::vector<std::string> const& cfg,
        hpx::runtime_mode mode)
    {
        hpx::init_params iparams;
        iparams.cfg = cfg;
        iparams.mode = mode;
        return start(f, detail::dummy_argc, detail::dummy_argv, iparams);
    }
#endif
}    // namespace hpx
