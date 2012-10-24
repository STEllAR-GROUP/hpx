//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_START_IMPL_OCT_04_2012_0252PM)
#define HPX_START_IMPL_OCT_04_2012_0252PM

#if !defined(HPX_START_OCT_04_2012_048PM)
#  error Do not directly include hpx/hpx_start_impl.hpp, use hpx/hpx_start.hpp instead!
#endif

namespace hpx
{
    /// \brief Main, non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is the main, non-blocking entry point for any HPX application.
    /// This function (or one of its overloads below) should be called from the
    /// users `main()` function. It will set up the HPX runtime environment and
    /// schedule the function given by \p f as a HPX thread. It will return
    /// immediatly after that. Use `hpx::wait` and `hpx::stop` to synchronize
    /// with the runtime system's execution.
    inline void
    start(int (*f)(boost::program_options::variables_map& vm),
        boost::program_options::options_description& desc_cmdline,
        int argc, char* argv[], HPX_STD_FUNCTION<void()> const& startup,
        HPX_STD_FUNCTION<void()> const& shutdown, hpx::runtime_mode mode)
    {
        std::vector<std::string> cfg;
        start(f, desc_cmdline, argc, argv, cfg, startup, shutdown, mode);
    }

    /// \brief Main, non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main, non-blocking entry point, which can be used
    /// to set up the runtime for an HPX application (the runtime system will be
    /// set up in console mode or worker mode depending on the command line
    /// settings). It will return immediatly after that. Use `hpx::wait` and
    /// `hpx::stop` to synchronize with the runtime system's execution.
    inline void
    start(boost::program_options::options_description& desc_cmdline,
        int argc, char* argv[], HPX_STD_FUNCTION<void()> const& startup,
        HPX_STD_FUNCTION<void()> const& shutdown, hpx::runtime_mode mode)
    {
        start(static_cast<hpx_main_type>(::hpx_main), desc_cmdline,
            argc, argv, startup, shutdown, mode);
    }

    /// \brief Main, non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main, non-blocking entry point, which can be used
    /// to set up the runtime for an HPX application (the runtime system will
    /// be set up in console mode or worker mode depending on the command line
    /// settings). It will return immediatly after that. Use `hpx::wait` and
    /// `hpx::stop` to synchronize with the runtime system's execution.
    inline void
    start(boost::program_options::options_description& desc_cmdline,
        int argc, char* argv[], std::vector<std::string> const& cfg,
        HPX_STD_FUNCTION<void()> const& startup,
        HPX_STD_FUNCTION<void()> const& shutdown, hpx::runtime_mode mode)
    {
        start(static_cast<hpx_main_type>(::hpx_main), desc_cmdline,
            argc, argv, cfg, startup, shutdown, mode);
    }

    /// \brief Main, non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main, non-blocking entry point, which can be used
    /// to set up the runtime for an HPX application (the runtime system will
    /// be set up in console mode or worker mode depending on the command line
    /// settings). It will return immediatly after that. Use `hpx::wait` and
    /// `hpx::stop` to synchronize with the runtime system's execution.
    inline void
    start(boost::program_options::options_description& desc_cmdline, int argc,
        char* argv[], hpx::runtime_mode mode)
    {
        HPX_STD_FUNCTION<void()> const empty;
        start(static_cast<hpx_main_type>(::hpx_main), desc_cmdline,
            argc, argv, empty, empty, mode);
    }

    /// \brief Main, non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main, non-blocking entry point, which can be used
    /// to set up the runtime for an HPX application (the runtime system will
    /// be set up in console mode or worker mode depending on the command line
    /// settings). It will return immediatly after that. Use `hpx::wait` and
    /// `hpx::stop` to synchronize with the runtime system's execution.
    inline void
    start(std::string const& app_name, int argc, char* argv[])
    {
        start(static_cast<hpx_main_type>(::hpx_main), app_name, argc, argv);
    }

    /// \brief Main, non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main, non-blocking entry point, which can be used
    /// to set up the runtime for an HPX application (the runtime system will
    /// be set up in console mode or worker mode depending on the command line
    /// settings). It will return immediatly after that. Use `hpx::wait` and
    /// `hpx::stop` to synchronize with the runtime system's execution.
    inline void start(int argc, char* argv[])
    {
        start(static_cast<hpx_main_type>(::hpx_main),
            HPX_APPLICATION_STRING, argc, argv);
    }

    /// \brief Main, non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main, non-blocking entry point, which can be used
    /// to set up the runtime for an HPX application (the runtime system will
    /// be set up in console mode or worker mode depending on the command line
    /// settings). It will return immediatly after that. Use `hpx::wait` and
    /// `hpx::stop` to synchronize with the runtime system's execution.
    inline void start(int (*f)(boost::program_options::variables_map& vm),
        std::string const& app_name, int argc, char* argv[])
    {
        using boost::program_options::options_description;

        options_description desc_commandline(
            "Usage: " + app_name +  " [options]");

        if (argc == 0 || argv == 0)
        {
            char *dummy_argv[1] = { const_cast<char*>(app_name.c_str()) };
            start(desc_commandline, 1, dummy_argv);
            return;
        }

        HPX_STD_FUNCTION<void()> const empty;
        start(f, desc_commandline, argc, argv, empty, empty);
    }
}

#endif
