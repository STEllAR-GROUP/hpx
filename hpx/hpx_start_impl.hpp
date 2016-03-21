//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_START_IMPL_OCT_04_2012_0252PM)
#define HPX_START_IMPL_OCT_04_2012_0252PM

#include <hpx/hpx_start.hpp>

#include <hpx/util/find_prefix.hpp>

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
            int argc, char** argv, std::vector<std::string> && ini_config,
            startup_function_type const& startup,
            shutdown_function_type const& shutdown, hpx::runtime_mode mode,
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
    /// schedule the function given by \p f as a HPX thread. It will return
    /// immediatly after that. Use `hpx::wait` and `hpx::stop` to synchronize
    /// with the runtime system's execution.
    inline bool start(
        util::function_nonser<
            int(boost::program_options::variables_map& vm)
        > const& f,
        boost::program_options::options_description const& desc_cmdline,
        int argc, char** argv, std::vector<std::string> const& cfg,
        util::function_nonser<void()> const& startup,
        util::function_nonser<void()> const& shutdown,
        hpx::runtime_mode mode)
    {
#if defined(HPX_WINDOWS)
        detail::init_winsocket();
#endif
        util::set_hpx_prefix(HPX_PREFIX);
        return 0 == detail::run_or_start(f, desc_cmdline, argc, argv,
            hpx_startup::user_main_config(cfg),
            startup, shutdown, mode, false);
    }

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is the main, non-blocking entry point for any HPX application.
    /// This function (or one of its overloads below) should be called from the
    /// users `main()` function. It will set up the HPX runtime environment and
    /// schedule the function given by \p f as a HPX thread. It will return
    /// immediatly after that. Use `hpx::wait` and `hpx::stop` to synchronize
    /// with the runtime system's execution.
    inline bool
    start(int (*f)(boost::program_options::variables_map& vm),
        boost::program_options::options_description const& desc_cmdline,
        int argc, char** argv, util::function_nonser<void()> const& startup,
        util::function_nonser<void()> const& shutdown, hpx::runtime_mode mode)
    {
        std::vector<std::string> cfg;
        return start(f, desc_cmdline, argc, argv, cfg, startup, shutdown, mode);
    }

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main, non-blocking entry point, which can be used
    /// to set up the runtime for an HPX application (the runtime system will be
    /// set up in console mode or worker mode depending on the command line
    /// settings). It will return immediatly after that. Use `hpx::wait` and
    /// `hpx::stop` to synchronize with the runtime system's execution.
    inline bool
    start(boost::program_options::options_description const& desc_cmdline,
        int argc, char** argv, util::function_nonser<void()> const& startup,
        util::function_nonser<void()> const& shutdown, hpx::runtime_mode mode)
    {
        return start(static_cast<hpx_main_type>(::hpx_main), desc_cmdline,
            argc, argv, startup, shutdown, mode);
    }

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main, non-blocking entry point, which can be used
    /// to set up the runtime for an HPX application (the runtime system will
    /// be set up in console mode or worker mode depending on the command line
    /// settings). It will return immediatly after that. Use `hpx::wait` and
    /// `hpx::stop` to synchronize with the runtime system's execution.
    inline bool
    start(boost::program_options::options_description const& desc_cmdline,
        int argc, char** argv, std::vector<std::string> const& cfg,
        util::function_nonser<void()> const& startup,
        util::function_nonser<void()> const& shutdown, hpx::runtime_mode mode)
    {
        return start(static_cast<hpx_main_type>(::hpx_main), desc_cmdline,
            argc, argv, cfg, startup, shutdown, mode);
    }

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main, non-blocking entry point, which can be used
    /// to set up the runtime for an HPX application (the runtime system will
    /// be set up in console mode or worker mode depending on the command line
    /// settings). It will return immediatly after that. Use `hpx::wait` and
    /// `hpx::stop` to synchronize with the runtime system's execution.
    inline bool
    start(int argc, char** argv, std::vector<std::string> const& cfg,
        hpx::runtime_mode mode)
    {
        using boost::program_options::options_description;

        options_description desc_commandline(
            "Usage: " HPX_APPLICATION_STRING " [options]");

        util::function_nonser<void()> const empty;
        return start(desc_commandline, argc, argv, cfg, empty, empty, mode);
    }

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main, non-blocking entry point, which can be used
    /// to set up the runtime for an HPX application (the runtime system will
    /// be set up in console mode or worker mode depending on the command line
    /// settings). It will return immediatly after that. Use `hpx::wait` and
    /// `hpx::stop` to synchronize with the runtime system's execution.
    inline bool
    start(boost::program_options::options_description const& desc_cmdline,
        int argc, char** argv, hpx::runtime_mode mode)
    {
        util::function_nonser<void()> const empty;
        return start(static_cast<hpx_main_type>(::hpx_main), desc_cmdline,
            argc, argv, empty, empty, mode);
    }

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main, non-blocking entry point, which can be used
    /// to set up the runtime for an HPX application (the runtime system will
    /// be set up in console mode or worker mode depending on the command line
    /// settings). It will return immediatly after that. Use `hpx::wait` and
    /// `hpx::stop` to synchronize with the runtime system's execution.
    inline bool
    start(boost::program_options::options_description const& desc_cmdline,
        int argc, char** argv, std::vector<std::string> const& cfg,
        hpx::runtime_mode mode)
    {
        util::function_nonser<void()> const empty;
        return start(desc_cmdline, argc, argv, cfg, empty, empty, mode);
    }

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main, non-blocking entry point, which can be used
    /// to set up the runtime for an HPX application (the runtime system will
    /// be set up in console mode or worker mode depending on the command line
    /// settings). It will return immediatly after that. Use `hpx::wait` and
    /// `hpx::stop` to synchronize with the runtime system's execution.
    inline bool
    start(std::string const& app_name, int argc, char** argv,
        hpx::runtime_mode mode)
    {
        util::function_nonser<void()> const empty;
        return start(static_cast<hpx_main_type>(::hpx_main), app_name, argc,
            argv, empty, empty, mode);
    }

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main, non-blocking entry point, which can be used
    /// to set up the runtime for an HPX application (the runtime system will
    /// be set up in console mode or worker mode depending on the command line
    /// settings). It will return immediatly after that. Use `hpx::wait` and
    /// `hpx::stop` to synchronize with the runtime system's execution.
    inline bool start(int argc, char** argv, hpx::runtime_mode mode)
    {
        return start(static_cast<hpx_main_type>(::hpx_main),
            HPX_APPLICATION_STRING, argc, argv, mode);
    }

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main, non-blocking entry point, which can be used
    /// to set up the runtime for an HPX application (the runtime system will
    /// be set up in console mode or worker mode depending on the command line
    /// settings). It will return immediatly after that. Use `hpx::wait` and
    /// `hpx::stop` to synchronize with the runtime system's execution.
    inline bool start(std::vector<std::string> const& cfg,
        hpx::runtime_mode mode)
    {
        using boost::program_options::options_description;

        options_description desc_commandline(
            std::string("Usage: ") + HPX_APPLICATION_STRING +  " [options]");

        char *dummy_argv[2] = { const_cast<char*>(HPX_APPLICATION_STRING), 0 };

        util::function_nonser<void()> const empty;
        return start(static_cast<hpx_main_type>(::hpx_main), desc_commandline,
            1, dummy_argv, cfg, empty, empty, mode);
    }

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main, non-blocking entry point, which can be used
    /// to set up the runtime for an HPX application (the runtime system will
    /// be set up in console mode or worker mode depending on the command line
    /// settings). It will return immediatly after that. Use `hpx::wait` and
    /// `hpx::stop` to synchronize with the runtime system's execution.
    inline bool start(int (*f)(boost::program_options::variables_map& vm),
        std::string const& app_name, int argc, char** argv,
        hpx::runtime_mode mode)
    {
        using boost::program_options::options_description;

        options_description desc_commandline(
            "Usage: " + app_name +  " [options]");

        if (argc == 0 || argv == 0)
        {
            char *dummy_argv[2] = { const_cast<char*>(app_name.c_str()), 0 };
            return start(desc_commandline, 1, dummy_argv, mode);
        }

        util::function_nonser<void()> const empty;
        return start(f, desc_commandline, argc, argv, empty, empty, mode);
    }

    // Main non-blocking entry point for launching the HPX runtime system.
    inline bool start(int (*f)(boost::program_options::variables_map& vm),
        int argc, char** argv, hpx::runtime_mode mode)
    {
        std::string app_name(HPX_APPLICATION_STRING);
        return start(f, app_name, argc, argv, mode);
    }

    /// \cond NOINTERNAL
    namespace detail
    {
        HPX_EXPORT int init_helper(
            boost::program_options::variables_map&,
            util::function_nonser<int(int, char**)> const&);
    }
    /// \endcond

    // Main non-blocking entry point for launching the HPX runtime system.
    inline bool start(util::function_nonser<int(int, char**)> const& f,
        std::string const& app_name, int argc, char** argv,
        hpx::runtime_mode mode)
    {
        using boost::program_options::options_description;

        options_description desc_commandline(
            "Usage: " + app_name +  " [options]");

        if (argc == 0 || argv == 0)
        {
            char *dummy_argv[2] = { const_cast<char*>(app_name.c_str()), 0 };
            return start(desc_commandline, 1, dummy_argv, mode);
        }

        std::vector<std::string> cfg;
        util::function_nonser<void()> const empty;

        return start(
            util::bind(detail::init_helper, util::placeholders::_1, f),
            desc_commandline, argc, argv, cfg, empty, empty, mode);
    }

    // Main non-blocking entry point for launching the HPX runtime system.
    inline bool start(util::function_nonser<int(int, char**)> const& f,
        int argc, char** argv, hpx::runtime_mode mode)
    {
        std::string app_name(HPX_APPLICATION_STRING);
        return start(f, app_name, argc, argv, mode);
    }
}

#endif
