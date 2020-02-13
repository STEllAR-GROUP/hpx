//  Copyright (c)      2018 Mikael Simberg
//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_HPX_START_IMPL_HPP
#define HPX_HPX_START_IMPL_HPP

#include <hpx/assertion.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/hpx_user_main_config.hpp>
#include <hpx/program_options.hpp>
#include <hpx/runtime_configuration/runtime_mode.hpp>
#include <hpx/runtime/shutdown_function.hpp>
#include <hpx/runtime/startup_function.hpp>
#include <hpx/functional/bind_back.hpp>
#include <hpx/prefix/find_prefix.hpp>
#include <hpx/functional/function.hpp>

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

namespace hpx
{
    /// \cond NOINTERNAL
    namespace detail
    {
        HPX_EXPORT int run_or_start(
            util::function_nonser<
                int(hpx::program_options::variables_map& vm)
            > const& f,
            hpx::program_options::options_description const& desc_cmdline,
            int argc, char** argv, std::vector<std::string>&& ini_config,
            startup_function_type startup, shutdown_function_type shutdown,
            hpx::runtime_mode mode, bool blocking);

        HPX_EXPORT int run_or_start(resource::partitioner& rp,
            startup_function_type startup, shutdown_function_type shutdown,
            bool blocking);

#if defined(HPX_WINDOWS)
        void init_winsocket();
#endif
    }
    /// \endcond

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is the main, non-blocking entry point for any HPX application.
    /// This function (or one of its overloads below) should be called from the
    /// users `main()` function. It will set up the HPX runtime environment and
    /// schedule the function given by \p f as an HPX thread. It will return
    /// immediately after that. Use `hpx::wait` and `hpx::stop` to synchronize
    /// with the runtime system's execution.
    inline bool start(hpx::init_params& params)
    {
#if defined(HPX_WINDOWS)
        detail::init_winsocket();
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

        return 0 == detail::run_or_start(params.f, (*params.desc_cmdline_ptr),
            params.argc, params.argv, hpx_startup::user_main_config(params.cfg),
            std::move(params.startup), std::move(params.shutdown), params.mode,
            false);
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
        util::function_nonser<
            int(hpx::program_options::variables_map& vm)
        > const& f,
        hpx::program_options::options_description const& desc_cmdline,
        int argc, char** argv, std::vector<std::string> const& cfg,
        startup_function_type startup, shutdown_function_type shutdown,
        hpx::runtime_mode mode)
    {
        using hpx::program_options::options_description;
        hpx::init_params iparams;
        iparams.f = f;
        iparams.desc_cmdline_ptr = std::make_shared<options_description>(
            desc_cmdline);
        iparams.argc = argc;
        iparams.argv = argv;
        iparams.cfg = cfg;
        iparams.startup = std::move(startup);
        iparams.shutdown = std::move(shutdown);
        iparams.mode = mode;
        return start(iparams);
    }

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is the main, non-blocking entry point for any HPX application.
    /// This function (or one of its overloads below) should be called from the
    /// users `main()` function. It will set up the HPX runtime environment and
    /// schedule the function given by \p f as an HPX thread. It will return
    /// immediately after that. Use `hpx::wait` and `hpx::stop` to synchronize
    /// with the runtime system's execution.
    inline bool
    start(int (*f)(hpx::program_options::variables_map& vm),
        hpx::program_options::options_description const& desc_cmdline,
        int argc, char** argv, startup_function_type startup,
        shutdown_function_type shutdown, hpx::runtime_mode mode)
    {
        std::vector<std::string> cfg;
        using hpx::program_options::options_description;

        hpx::init_params iparams;
        iparams.f = f;
        iparams.desc_cmdline_ptr = std::make_shared<options_description>(
            desc_cmdline);
        iparams.argc = argc;
        iparams.argv = argv;
        iparams.cfg = cfg;
        iparams.startup = std::move(startup);
        iparams.shutdown = std::move(shutdown);
        iparams.mode = mode;
        return start(iparams);
    }

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main, non-blocking entry point, which can be used
    /// to set up the runtime for an HPX application (the runtime system will be
    /// set up in console mode or worker mode depending on the command line
    /// settings). It will return immediately after that. Use `hpx::wait` and
    /// `hpx::stop` to synchronize with the runtime system's execution.
    inline bool
    start(hpx::program_options::options_description const& desc_cmdline,
        int argc, char** argv, startup_function_type startup,
        shutdown_function_type shutdown, hpx::runtime_mode mode)
    {
        using hpx::program_options::options_description;

        hpx::init_params iparams;
        iparams.f = static_cast<hpx_main_type>(::hpx_main);
        iparams.desc_cmdline_ptr = std::make_shared<options_description>(
            desc_cmdline);
        iparams.argc = argc;
        iparams.argv = argv;
        iparams.startup = std::move(startup);
        iparams.shutdown = std::move(shutdown);
        iparams.mode = mode;
        return start(iparams);
    }

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main, non-blocking entry point, which can be used
    /// to set up the runtime for an HPX application (the runtime system will
    /// be set up in console mode or worker mode depending on the command line
    /// settings). It will return immediately after that. Use `hpx::wait` and
    /// `hpx::stop` to synchronize with the runtime system's execution.
    inline bool
    start(hpx::program_options::options_description const& desc_cmdline,
        int argc, char** argv, std::vector<std::string> const& cfg,
        startup_function_type startup, shutdown_function_type shutdown,
        hpx::runtime_mode mode)
    {
        using hpx::program_options::options_description;

        hpx::init_params iparams;
        iparams.f = static_cast<hpx_main_type>(::hpx_main);
        iparams.desc_cmdline_ptr = std::make_shared<options_description>(
            desc_cmdline);
        iparams.argc = argc;
        iparams.argv = argv;
        iparams.cfg = cfg;
        iparams.startup = std::move(startup);
        iparams.shutdown = std::move(shutdown);
        iparams.mode = mode;
        return start(iparams);
    }

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main, non-blocking entry point, which can be used
    /// to set up the runtime for an HPX application (the runtime system will
    /// be set up in console mode or worker mode depending on the command line
    /// settings). It will return immediately after that. Use `hpx::wait` and
    /// `hpx::stop` to synchronize with the runtime system's execution.
    inline bool
    start(int argc, char** argv, std::vector<std::string> const& cfg,
        hpx::runtime_mode mode)
    {
        hpx::init_params iparams;
        iparams.f = static_cast<hpx_main_type>(::hpx_main);
        iparams.argc = argc;
        iparams.argv = argv;
        iparams.cfg = cfg;
        iparams.mode = mode;
        return start(iparams);
    }

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main, non-blocking entry point, which can be used
    /// to set up the runtime for an HPX application (the runtime system will
    /// be set up in console mode or worker mode depending on the command line
    /// settings). It will return immediately after that. Use `hpx::wait` and
    /// `hpx::stop` to synchronize with the runtime system's execution.
    inline bool
    start(hpx::program_options::options_description const& desc_cmdline,
        int argc, char** argv, hpx::runtime_mode mode)
    {
        using hpx::program_options::options_description;

        hpx::init_params iparams;
        iparams.f = static_cast<hpx_main_type>(::hpx_main);
        iparams.desc_cmdline_ptr = std::make_shared<options_description>(
            desc_cmdline);
        iparams.argc = argc;
        iparams.argv = argv;
        iparams.mode = mode;
        return start(iparams);
    }

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main, non-blocking entry point, which can be used
    /// to set up the runtime for an HPX application (the runtime system will
    /// be set up in console mode or worker mode depending on the command line
    /// settings). It will return immediately after that. Use `hpx::wait` and
    /// `hpx::stop` to synchronize with the runtime system's execution.
    inline bool
    start(hpx::program_options::options_description const& desc_cmdline,
        int argc, char** argv, std::vector<std::string> const& cfg,
        hpx::runtime_mode mode)
    {
        using hpx::program_options::options_description;

        hpx::init_params iparams;
        iparams.f = static_cast<hpx_main_type>(::hpx_main);
        iparams.desc_cmdline_ptr = std::make_shared<options_description>(
            desc_cmdline);
        iparams.argc = argc;
        iparams.argv = argv;
        iparams.cfg = cfg;
        iparams.mode = mode;
        return start(iparams);
    }

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main, non-blocking entry point, which can be used
    /// to set up the runtime for an HPX application (the runtime system will
    /// be set up in console mode or worker mode depending on the command line
    /// settings). It will return immediately after that. Use `hpx::wait` and
    /// `hpx::stop` to synchronize with the runtime system's execution.
    inline bool
    start(std::string const& app_name, int argc, char** argv,
        hpx::runtime_mode mode)
    {
        using hpx::program_options::options_description;
        options_description desc = options_description("Usage: " + app_name +
                " [options]");
        hpx::init_params iparams;
        iparams.f = static_cast<hpx_main_type>(::hpx_main);
        iparams.desc_cmdline_ptr = std::make_shared<options_description>(desc);
        iparams.argc = argc;
        iparams.argv = argv;
        iparams.mode = mode;
        return start(iparams);
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
        iparams.f = static_cast<hpx_main_type>(::hpx_main);
        iparams.argc = argc;
        iparams.argv = argv;
        iparams.mode = mode;
        return start(iparams);
    }

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main, non-blocking entry point, which can be used
    /// to set up the runtime for an HPX application (the runtime system will
    /// be set up in console mode or worker mode depending on the command line
    /// settings). It will return immediately after that. Use `hpx::wait` and
    /// `hpx::stop` to synchronize with the runtime system's execution.
    inline bool start(std::vector<std::string> const& cfg,
        hpx::runtime_mode mode)
    {
        hpx::init_params iparams;
        iparams.f = static_cast<hpx_main_type>(::hpx_main);
        iparams.cfg = cfg;
        iparams.mode = mode;
        return start(iparams);
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
            "Usage: " + app_name +  " [options]");

        hpx::init_params iparams;
        iparams.desc_cmdline_ptr = std::make_shared<options_description>(
            desc_commandline);
        iparams.mode = mode;

        if (argc == 0 || argv == nullptr)
        {
            iparams.f = static_cast<hpx_main_type>(::hpx_main);
            return start(iparams);
        }

        iparams.f = f;
        iparams.argc = argc;
        iparams.argv = argv;
        return start(iparams);
    }

    // Main non-blocking entry point for launching the HPX runtime system.
    inline bool start(int (*f)(hpx::program_options::variables_map& vm),
        int argc, char** argv, hpx::runtime_mode mode)
    {
        hpx::init_params iparams;

        if (argc == 0 || argv == nullptr)
        {
            iparams.f = static_cast<hpx_main_type>(::hpx_main);
            return start(iparams);
        }

        iparams.f = f;
        iparams.argc = argc;
        iparams.argv = argv;
        iparams.mode = mode;
        return start(iparams);
    }

    /// \cond NOINTERNAL
    namespace detail
    {
        HPX_EXPORT int init_helper(
            hpx::program_options::variables_map&,
            util::function_nonser<int(int, char**)> const&);
    }
    /// \endcond

    // Main non-blocking entry point for launching the HPX runtime system.
    inline bool start(util::function_nonser<int(int, char**)> const& f,
        std::string const& app_name, int argc, char** argv,
        hpx::runtime_mode mode)
    {
        using hpx::program_options::options_description;
        options_description desc_commandline(
            "Usage: " + app_name +  " [options]");

        HPX_ASSERT(argc != 0 && argv != nullptr);

        hpx::init_params iparams;
        iparams.f = util::bind_back(detail::init_helper, f);
        iparams.desc_cmdline_ptr = std::make_shared<options_description>(
            desc_commandline);
        iparams.argc = argc;
        iparams.argv = argv;
        iparams.mode = mode;
        return start(iparams);
    }

    // Main non-blocking entry point for launching the HPX runtime system.
    inline bool start(util::function_nonser<int(int, char**)> const& f,
        int argc, char** argv, hpx::runtime_mode mode)
    {
        HPX_ASSERT(argc != 0 && argv != nullptr);

        hpx::init_params iparams;
        iparams.f = util::bind_back(detail::init_helper, f);
        iparams.argc = argc;
        iparams.argv = argv;
        iparams.mode = mode;
        return start(iparams);
    }

    inline bool start(util::function_nonser<int(int, char**)> const& f,
        int argc, char** argv, std::vector<std::string> const& cfg,
        hpx::runtime_mode mode)
    {
        HPX_ASSERT(argc != 0 && argv != nullptr);

        hpx::init_params iparams;
        iparams.f = util::bind_back(detail::init_helper, f);
        iparams.argc = argc;
        iparams.argv = argv;
        iparams.cfg = cfg;
        iparams.mode = mode;
        return start(iparams);
    }

    inline bool start(util::function_nonser<int(int, char**)> const& f,
        std::vector<std::string> const& cfg,
        hpx::runtime_mode mode)
    {
        hpx::init_params iparams;
        iparams.f = util::bind_back(detail::init_helper, f);
        iparams.cfg = cfg;
        iparams.mode = mode;
        return start(iparams);
    }

    inline bool start(std::nullptr_t f, std::string const& app_name, int argc,
        char** argv, hpx::runtime_mode mode)
    {
        using hpx::program_options::options_description;
        options_description desc_commandline(
            "Usage: " + app_name +  " [options]");
        util::function_nonser<int(hpx::program_options::variables_map& vm)>
            main_f;

        HPX_ASSERT(argc != 0 && argv != nullptr);

        hpx::init_params iparams;
        iparams.f = main_f;
        iparams.desc_cmdline_ptr = std::make_shared<options_description>(
            desc_commandline);
        iparams.argc = argc;
        iparams.argv = argv;
        iparams.mode = mode;
        return start(iparams);
    }

    inline bool start(std::nullptr_t f, int argc, char** argv,
        hpx::runtime_mode mode)
    {
        util::function_nonser<int(hpx::program_options::variables_map& vm)>
            main_f;

        HPX_ASSERT(argc != 0 && argv != nullptr);

        hpx::init_params iparams;
        iparams.f = main_f;
        iparams.argc = argc;
        iparams.argv = argv;
        iparams.mode = mode;
        return start(iparams);
    }

    inline bool start(std::nullptr_t f, int argc, char** argv,
        std::vector<std::string> const& cfg, hpx::runtime_mode mode)
    {
        util::function_nonser<int(hpx::program_options::variables_map& vm)>
            main_f;

        HPX_ASSERT(argc != 0 && argv != nullptr);

        hpx::init_params iparams;
        iparams.f = main_f;
        iparams.argc = argc;
        iparams.argv = argv;
        iparams.cfg = cfg;
        iparams.mode = mode;
        return start(iparams);
    }

    inline bool start(std::nullptr_t f, std::vector<std::string> const& cfg,
        hpx::runtime_mode mode)
    {
        util::function_nonser<int(hpx::program_options::variables_map& vm)>
            main_f;

        hpx::init_params iparams;
        iparams.f = main_f;
        iparams.cfg = cfg;
        iparams.mode = mode;
        return start(iparams);
    }

    ////////////////////////////////////////////////////////////////////////////
    inline bool start(resource::partitioner& rp, startup_function_type startup,
        shutdown_function_type shutdown)
    {
#if defined(HPX_WINDOWS)
        detail::init_winsocket();
#endif
        util::set_hpx_prefix(HPX_PREFIX);
#if defined(__FreeBSD__)
        freebsd_environ = environ;
#endif
        // set a handler for std::abort
        std::signal(SIGABRT, detail::on_abort);
        std::atexit(detail::on_exit);
#if defined(HPX_HAVE_CXX11_STD_QUICK_EXIT)
        std::at_quick_exit(detail::on_exit);
#endif

        return 0 == detail::run_or_start(
            rp, std::move(startup), std::move(shutdown), false);
    }
}

#endif /*HPX_HPX_START_IMPL_HPP*/
