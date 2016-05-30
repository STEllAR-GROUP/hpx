//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_HPX_INIT_IMPL_HPP
#define HPX_HPX_INIT_IMPL_HPP

#include <hpx/hpx_init.hpp>
#include <hpx/hpx_user_main_config.hpp>
#include <hpx/runtime/runtime_mode.hpp>
#include <hpx/runtime/shutdown_function.hpp>
#include <hpx/runtime/startup_function.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/find_prefix.hpp>
#include <hpx/util/function.hpp>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>

#include <string>
#include <utility>
#include <vector>

namespace hpx
{
    /// \cond NOINTERNAL
    namespace detail
    {
        HPX_EXPORT int run_or_start(
            util::function_nonser<
                int(boost::program_options::variables_map& vm)
            > const& f,
            boost::program_options::options_description const& desc_cmdline,
            int argc, char** argv, std::vector<std::string>&& ini_config,
            startup_function_type startup, shutdown_function_type shutdown,
            hpx::runtime_mode mode, bool blocking);

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
    inline int init(
        util::function_nonser<
            int(boost::program_options::variables_map& vm)
        > const& f,
        boost::program_options::options_description const& desc_cmdline,
        int argc, char** argv, std::vector<std::string> const& cfg,
        startup_function_type startup, shutdown_function_type shutdown,
        hpx::runtime_mode mode)
    {
#if defined(HPX_WINDOWS)
        detail::init_winsocket();
#endif
        util::set_hpx_prefix(HPX_PREFIX);
        return detail::run_or_start(f, desc_cmdline, argc, argv,
            hpx_startup::user_main_config(cfg),
            std::move(startup), std::move(shutdown), mode, true);
    }

    /// \brief Main entry point for launching the HPX runtime system.
    ///
    /// This is the main entry point for any HPX application. This function
    /// (or one of its overloads below) should be called from the users `main()`
    /// function. It will set up the HPX runtime environment and schedule the
    /// function given by \p f as a HPX thread.
    inline int
    init(int (*f)(boost::program_options::variables_map& vm),
        boost::program_options::options_description const& desc_cmdline,
        int argc, char** argv, startup_function_type startup,
        shutdown_function_type shutdown, hpx::runtime_mode mode)
    {
        std::vector<std::string> cfg;
        return init(f, desc_cmdline, argc, argv, cfg, std::move(startup),
            std::move(shutdown), mode);
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
    init(boost::program_options::options_description const& desc_cmdline,
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
    init(boost::program_options::options_description const& desc_cmdline,
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
        using boost::program_options::options_description;

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
    init(boost::program_options::options_description const& desc_cmdline,
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
    init(boost::program_options::options_description const& desc_cmdline,
        int argc, char** argv, std::vector<std::string> const& cfg,
        hpx::runtime_mode mode)
    {
        return init(desc_cmdline, argc, argv, cfg, startup_function_type(),
            shutdown_function_type(), mode);
    }

    /// \fn int init(std::string const& app_name, int argc = 0, char** argv = 0)
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
        using boost::program_options::options_description;

        options_description desc_commandline(
            std::string("Usage: ") + HPX_APPLICATION_STRING +  " [options]");

        char *dummy_argv[2] = { const_cast<char*>(HPX_APPLICATION_STRING), 0 };

        return init(static_cast<hpx_main_type>(::hpx_main), desc_commandline,
            1, dummy_argv, cfg, startup_function_type(),
            shutdown_function_type(), mode);
    }

    /// \brief Main entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main entry point, which can be used to set up the
    /// runtime for an HPX application (the runtime system will be set up in
    /// console mode or worker mode depending on the command line settings).
    inline int init(int (*f)(boost::program_options::variables_map&),
        std::string const& app_name, int argc, char** argv,
        hpx::runtime_mode mode)
    {
        using boost::program_options::options_description;

        options_description desc_commandline(
            "Usage: " + app_name +  " [options]");

        if (argc == 0 || argv == 0)
        {
            char *dummy_argv[2] = { const_cast<char*>(app_name.c_str()), 0 };
            return init(desc_commandline, 1, dummy_argv, mode);
        }

        return init(f, desc_commandline, argc, argv, startup_function_type(),
            shutdown_function_type(), mode);
    }

    // Main entry point for launching the HPX runtime system.
    inline int init(int (*f)(boost::program_options::variables_map&),
        int argc, char** argv, hpx::runtime_mode mode)
    {
        std::string app_name(HPX_APPLICATION_STRING);
        return init(f, app_name, argc, argv, mode);
    }

    /// \cond NOINTERNAL
    namespace detail
    {
        HPX_EXPORT int init_helper(
            boost::program_options::variables_map&,
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
        using boost::program_options::options_description;
        options_description desc_commandline(
            std::string("Usage: ") + app_name +  " [options]");

        util::function_nonser<int(boost::program_options::variables_map& vm)>
            main_f = util::bind(detail::init_helper, util::placeholders::_1, f);
        std::vector<std::string> cfg;
        util::function_nonser<void()> const empty;

        HPX_ASSERT(argc != 0 && argv != 0);

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
        using boost::program_options::options_description;

        options_description desc_commandline(
            "Usage: " + app_name +  " [options]");

        util::function_nonser<int(boost::program_options::variables_map& vm)>
            main_f = util::bind(detail::init_helper, util::placeholders::_1, f);

        HPX_ASSERT(argc != 0 && argv != 0);

        return init(main_f, desc_commandline, argc, argv, cfg,
            startup_function_type(), shutdown_function_type(), mode);
    }
}

#endif /*HPX_HPX_INIT_IMPL_HPP*/
