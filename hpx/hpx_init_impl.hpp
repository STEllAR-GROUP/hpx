//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_INIT_IMPL_OCT_04_2012_0123PM)
#define HPX_INIT_IMPL_OCT_04_2012_0123PM

#if !defined(HPX_INIT_OCT_04_2012_0132PM)
#  error Do not directly include hpx/hpx_init_impl.hpp, use hpx/hpx_init.hpp instead!
#endif

#include <hpx/util/bind.hpp>

namespace hpx
{
    /// \brief Main entry point for launching the HPX runtime system.
    ///
    /// This is the main entry point for any HPX application. This function
    /// (or one of its overloads below) should be called from the users `main()`
    /// function. It will set up the HPX runtime environment and schedule the
    /// function given by \p f as a HPX thread.
    inline int
    init(int (*f)(boost::program_options::variables_map& vm),
        boost::program_options::options_description const& desc_cmdline,
        int argc, char** argv, HPX_STD_FUNCTION<void()> const& startup,
        HPX_STD_FUNCTION<void()> const& shutdown, hpx::runtime_mode mode)
    {
        std::vector<std::string> cfg;
        return init(f, desc_cmdline, argc, argv, cfg, startup, shutdown, mode);
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
        int argc, char** argv, HPX_STD_FUNCTION<void()> const& startup,
        HPX_STD_FUNCTION<void()> const& shutdown, hpx::runtime_mode mode)
    {
        return init(static_cast<hpx_main_type>(::hpx_main), desc_cmdline,
            argc, argv, startup, shutdown, mode);
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
        HPX_STD_FUNCTION<void()> const& startup,
        HPX_STD_FUNCTION<void()> const& shutdown, hpx::runtime_mode mode)
    {
        return init(static_cast<hpx_main_type>(::hpx_main), desc_cmdline,
            argc, argv, cfg, startup, shutdown, mode);
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

        HPX_STD_FUNCTION<void()> const empty;
        return init(desc_commandline, argc, argv, cfg, empty, empty, mode);
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
    init(boost::program_options::options_description const& desc_cmdline, int argc,
        char** argv, hpx::runtime_mode mode)
    {
        HPX_STD_FUNCTION<void()> const empty;
        return init(static_cast<hpx_main_type>(::hpx_main), desc_cmdline,
            argc, argv, empty, empty, mode);
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
        HPX_STD_FUNCTION<void()> const empty;
        return init(static_cast<hpx_main_type>(::hpx_main), app_name,
            argc, argv, empty, empty, mode);
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
        HPX_STD_FUNCTION<void()> const empty;

        return init(static_cast<hpx_main_type>(::hpx_main), desc_commandline,
            1, dummy_argv, cfg, empty, empty, mode);
    }

    /// \brief Main entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main entry point, which can be used to set up the
    /// runtime for an HPX application (the runtime system will be set up in
    /// console mode or worker mode depending on the command line settings).
    inline int init(int (*f)(boost::program_options::variables_map& vm),
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

        HPX_STD_FUNCTION<void()> const empty;
        return init(f, desc_commandline, argc, argv, empty, empty, mode);
    }

    /// \brief Main entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main entry point, which can be used to set up the
    /// runtime for an HPX application (the runtime system will be set up in
    /// console mode or worker mode depending on the command line settings).
    namespace detail
    {
        inline int init_helper(boost::program_options::variables_map& /*vm*/,
            HPX_STD_FUNCTION<int(int, char**)> const& f, int argc, char** argv)
        {
            return f(argc, argv);
        }
    }

    inline int init(HPX_STD_FUNCTION<int(int, char**)> const& f,
        std::string const& /*app_name*/, int argc, char** argv, hpx::runtime_mode mode)
    {
        using boost::program_options::options_description;
        options_description desc_commandline(
            std::string("Usage: ") + HPX_APPLICATION_STRING +  " [options]");

        std::vector<std::string> cfg;
        HPX_STD_FUNCTION<void()> const empty;

        return init(
            util::bind(detail::init_helper, util::placeholders::_1, f, argc, argv),
            desc_commandline, argc, argv, cfg, empty, empty, mode);
    }
}

#endif
