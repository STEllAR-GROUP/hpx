//  Copyright (c)      2018 Mikael Simberg
//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_HPX_INIT_IMPL_HPP
#define HPX_HPX_INIT_IMPL_HPP

#include <hpx/assertion.hpp>
#include <hpx/hpx_init.hpp>
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
#include <memory>
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

    /// \brief Main entry point for launching the HPX runtime system.
    ///
    /// This is the main entry point for any HPX application. This function
    /// (or one of its overloads below) should be called from the users `main()`
    /// function. It will set up the HPX runtime environment and schedule the
    /// function given by \p f as a HPX thread.
    inline int init(hpx::init_params& params)
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

        return detail::run_or_start(params.f, (*params.desc_cmdline_ptr),
            params.argc, params.argv, hpx_startup::user_main_config(params.cfg),
            std::move(params.startup), std::move(params.shutdown), params.mode,
            true);
    }

    /// \brief Main entry point for launching the HPX runtime system.
    ///
    /// This is the main entry point for any HPX application. This function
    /// (or one of its overloads below) should be called from the users `main()`
    /// function. It will set up the HPX runtime environment and schedule the
    /// function given by \p f as a HPX thread.
    inline int init(
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
        return init(iparams);
    }

    /// \brief Main entry point for launching the HPX runtime system.
    ///
    /// This is the main entry point for any HPX application. This function
    /// (or one of its overloads below) should be called from the users `main()`
    /// function. It will set up the HPX runtime environment and schedule the
    /// function given by \p f as a HPX thread.
    inline int
    init(int (*f)(hpx::program_options::variables_map& vm),
        hpx::program_options::options_description const& desc_cmdline,
        int argc, char** argv, startup_function_type startup,
        shutdown_function_type shutdown, hpx::runtime_mode mode)
    {
        using options_desc_type = hpx::program_options::options_description;

        hpx::init_params iparams;
        iparams.f = f;
        iparams.desc_cmdline_ptr = std::make_shared<options_desc_type>(
            desc_cmdline);
        iparams.argc = argc;
        iparams.argv = argv;
        iparams.startup = std::move(startup);
        iparams.shutdown = std::move(shutdown);
        iparams.mode = mode;
        return init(iparams);
    }

    /// \brief Main entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main entry point, which can be used to set up the
    /// runtime for an HPX application (the runtime system will be set up in
    /// console mode or worker mode depending on the command line settings).
    ///
    /// In console mode it will execute the user supplied function `hpx_main`,
    /// in worker mode it will execute an empty `hpx_main`.
    inline int
    init(hpx::program_options::options_description const& desc_cmdline,
        int argc, char** argv, startup_function_type startup,
        shutdown_function_type shutdown, hpx::runtime_mode mode)
    {
        return init(static_cast<hpx_main_type>(::hpx_main), desc_cmdline,
            argc, argv, std::move(startup), std::move(shutdown), mode);
    }

    /// \brief Main entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main entry point, which can be used to set up the
    /// runtime for an HPX application (the runtime system will be set up in
    /// console mode or worker mode depending on the command line settings).
    ///
    /// In console mode it will execute the user supplied function `hpx_main`,
    /// in worker mode it will execute an empty `hpx_main`.
    inline int
    init(hpx::program_options::options_description const& desc_cmdline,
        int argc, char** argv, std::vector<std::string> const& cfg,
        startup_function_type startup, shutdown_function_type shutdown,
        hpx::runtime_mode mode)
    {
        return init(static_cast<hpx_main_type>(::hpx_main), desc_cmdline,
            argc, argv, cfg, std::move(startup), std::move(shutdown), mode);
    }

    /// \brief Main entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main entry point, which can be used to set up the
    /// runtime for an HPX application (the runtime system will be set up in
    /// console mode or worker mode depending on the command line settings).
    ///
    /// In console mode it will execute the user supplied function `hpx_main`,
    /// in worker mode it will execute an empty `hpx_main`.
    inline int
    init(int argc, char** argv, std::vector<std::string> const& cfg,
        hpx::runtime_mode mode)
    {
        using hpx::program_options::options_description;

        options_description desc_commandline(
            "Usage: " HPX_APPLICATION_STRING " [options]");

        return init(desc_commandline, argc, argv, cfg, startup_function_type(),
            shutdown_function_type(), mode);
    }

    /// \brief Main entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main entry point, which can be used to set up the
    /// runtime for an HPX application (the runtime system will be set up in
    /// console mode or worker mode depending on the command line settings).
    ///
    /// In console mode it will execute the user supplied function `hpx_main`,
    /// in worker mode it will execute an empty `hpx_main`.
    inline int
    init(hpx::program_options::options_description const& desc_cmdline,
        int argc, char** argv, hpx::runtime_mode mode)
    {
        return init(static_cast<hpx_main_type>(::hpx_main), desc_cmdline,
            argc, argv, startup_function_type(), shutdown_function_type(),
            mode);
    }

    /// \brief Main entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main entry point, which can be used to set up the
    /// runtime for an HPX application (the runtime system will be set up in
    /// console mode or worker mode depending on the command line settings).
    ///
    /// In console mode it will execute the user supplied function `hpx_main`,
    /// in worker mode it will execute an empty `hpx_main`.
    inline int
    init(hpx::program_options::options_description const& desc_cmdline,
        int argc, char** argv, std::vector<std::string> const& cfg,
        hpx::runtime_mode mode)
    {
        return init(desc_cmdline, argc, argv, cfg, startup_function_type(),
            shutdown_function_type(), mode);
    }

    /// \fn int init(std::string const& app_name, int argc = 0, char** argv = nullptr)
    ///
    /// \brief Main entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main entry point, which can be used to set up the
    /// runtime for an HPX application (the runtime system will be set up in
    /// console mode or worker mode depending on the command line settings).
    inline int
    init(std::string const& app_name, int argc, char** argv,
        hpx::runtime_mode mode)
    {
        return init(static_cast<hpx_main_type>(::hpx_main), app_name,
            argc, argv, startup_function_type(), shutdown_function_type(),
            mode);
    }

    /// \brief Main entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main entry point, which can be used to set up the
    /// runtime for an HPX application (the runtime system will be set up in
    /// console mode or worker mode depending on the command line settings).
    inline int init(int argc, char** argv, hpx::runtime_mode mode)
    {
        return init(static_cast<hpx_main_type>(::hpx_main),
            HPX_APPLICATION_STRING, argc, argv, mode);
    }

    /// \brief Main entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main entry point, which can be used to set up the
    /// runtime for an HPX application (the runtime system will be set up in
    /// console mode or worker mode depending on the command line settings).
    inline int init(std::vector<std::string> const& cfg,
        hpx::runtime_mode mode)
    {
        using hpx::program_options::options_description;

        options_description desc_commandline(
            std::string("Usage: ") + HPX_APPLICATION_STRING +  " [options]");

        char *dummy_argv[2] = { const_cast<char*>(HPX_APPLICATION_STRING), nullptr };

        return init(static_cast<hpx_main_type>(::hpx_main), desc_commandline,
            1, dummy_argv, cfg, startup_function_type(),
            shutdown_function_type(), mode);
    }

    /// \brief Main entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main entry point, which can be used to set up the
    /// runtime for an HPX application (the runtime system will be set up in
    /// console mode or worker mode depending on the command line settings).
    inline int init(int (*f)(hpx::program_options::variables_map&),
        std::string const& app_name, int argc, char** argv,
        hpx::runtime_mode mode)
    {
        using hpx::program_options::options_description;

        options_description desc_commandline(
            "Usage: " + app_name +  " [options]");

        if (argc == 0 || argv == nullptr)
        {
            char *dummy_argv[2] = { const_cast<char*>(app_name.c_str()), nullptr };
            return init(desc_commandline, 1, dummy_argv, mode);
        }

        return init(f, desc_commandline, argc, argv, startup_function_type(),
            shutdown_function_type(), mode);
    }

    // Main entry point for launching the HPX runtime system.
    inline int init(int (*f)(hpx::program_options::variables_map&),
        int argc, char** argv, hpx::runtime_mode mode)
    {
        std::string app_name(HPX_APPLICATION_STRING);
        return init(f, app_name, argc, argv, mode);
    }

    /// \cond NOINTERNAL
    namespace detail
    {
        HPX_EXPORT int init_helper(
            hpx::program_options::variables_map&,
            util::function_nonser<int(int, char**)> const&);
    }
    /// \endcond

    /// \brief Main entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main entry point, which can be used to set up the
    /// runtime for an HPX application (the runtime system will be set up in
    /// console mode or worker mode depending on the command line settings).
    inline int
    init(util::function_nonser<int(int, char**)> const& f,
        std::string const& app_name, int argc, char** argv,
        hpx::runtime_mode mode)
    {
        using hpx::program_options::options_description;
        options_description desc_commandline(
            std::string("Usage: ") + app_name +  " [options]");

        util::function_nonser<int(hpx::program_options::variables_map& vm)>
            main_f = util::bind_back(detail::init_helper, f);
        std::vector<std::string> cfg;
        util::function_nonser<void()> const empty;

        HPX_ASSERT(argc != 0 && argv != nullptr);

        return init(main_f, desc_commandline, argc, argv, cfg,
            empty, empty, mode);
    }

    // Main entry point for launching the HPX runtime system.
    inline int
    init(util::function_nonser<int(int, char**)> const& f,
        int argc, char** argv, hpx::runtime_mode mode)
    {
        std::string app_name(HPX_APPLICATION_STRING);
        return init(f, app_name, argc, argv, mode);
    }

    inline int
    init(util::function_nonser<int(int, char**)> const& f,
        int argc, char** argv, std::vector<std::string> const& cfg,
        hpx::runtime_mode mode)
    {
        std::string app_name(HPX_APPLICATION_STRING);
        using hpx::program_options::options_description;

        options_description desc_commandline(
            "Usage: " + app_name +  " [options]");

        util::function_nonser<int(hpx::program_options::variables_map& vm)>
            main_f = util::bind_back(detail::init_helper, f);

        HPX_ASSERT(argc != 0 && argv != nullptr);

        return init(main_f, desc_commandline, argc, argv, cfg,
            startup_function_type(), shutdown_function_type(), mode);
    }

    inline int
        init(util::function_nonser<int(int, char**)> const& f,
        std::vector<std::string> const& cfg,
        hpx::runtime_mode mode)
    {
        char *dummy_argv[2] = { const_cast<char*>(HPX_APPLICATION_STRING), nullptr };

        return init(f, 1, dummy_argv, cfg, mode);
    }

    inline int
    init(std::nullptr_t, std::string const& app_name, int argc, char** argv,
        hpx::runtime_mode mode)
    {
        using hpx::program_options::options_description;
        options_description desc_commandline(
            std::string("Usage: ") + app_name +  " [options]");

        util::function_nonser<int(hpx::program_options::variables_map& vm)>
            main_f;
        std::vector<std::string> cfg;
        util::function_nonser<void()> const empty;

        HPX_ASSERT(argc != 0 && argv != nullptr);

        return init(main_f, desc_commandline, argc, argv, cfg,
            empty, empty, mode);
    }

    inline int
    init(std::nullptr_t const& f,
        int argc, char** argv, hpx::runtime_mode mode)
    {
        std::string app_name(HPX_APPLICATION_STRING);
        return init(f, app_name, argc, argv, mode);
    }

    inline int
    init(std::nullptr_t,
        int argc, char** argv, std::vector<std::string> const& cfg,
        hpx::runtime_mode mode)
    {
        std::string app_name(HPX_APPLICATION_STRING);
        using hpx::program_options::options_description;

        options_description desc_commandline(
            "Usage: " + app_name +  " [options]");

        util::function_nonser<int(hpx::program_options::variables_map& vm)>
            main_f;

        HPX_ASSERT(argc != 0 && argv != nullptr);

        return init(main_f, desc_commandline, argc, argv, cfg,
            startup_function_type(), shutdown_function_type(), mode);
    }

    inline int
    init(std::nullptr_t, std::vector<std::string> const& cfg,
         hpx::runtime_mode mode)
    {
        char* dummy_argv[2] = {
            const_cast<char*>(HPX_APPLICATION_STRING), nullptr};

        return init(nullptr, 1, dummy_argv, cfg, mode);
    }

    ////////////////////////////////////////////////////////////////////////////
    inline int init(resource::partitioner& rp, startup_function_type startup,
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

        return detail::run_or_start(
            rp, std::move(startup), std::move(shutdown), true);
    }
}

#endif /*HPX_HPX_INIT_IMPL_HPP*/
