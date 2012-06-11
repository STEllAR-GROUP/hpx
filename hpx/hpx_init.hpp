//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2010-2011 Phillip LeBlanc, Dylan Stark
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// \file hpx_init.hpp

#if !defined(ABC9B037_3A25_4591_BB60_CD166773D61D)
#define ABC9B037_3A25_4591_BB60_CD166773D61D

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime.hpp>

#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>

/// \cond NOINTERNAL

///////////////////////////////////////////////////////////////////////////////
// One of these functions must be implemented by the application for the
// console locality.
int hpx_main();
int hpx_main(boost::program_options::variables_map& vm);

/// \endcond

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    /// \cond NOINTERNAL

#if !defined(HPX_NO_DEPRECATED)
    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    HPX_DEPRECATED(HPX_DEPRECATED_MSG)
    inline void
    get_option(boost::program_options::variables_map& vm,
               std::string const& name, T& x,
               std::string const& config = "")
    {
        if (vm.count(name))
            x = vm[name].as<T>();

        if (!config.empty()) {
            x = boost::lexical_cast<T>(
                get_runtime().get_config().get_entry(config, x));
        }
    }

    template <typename T>
    HPX_DEPRECATED(HPX_DEPRECATED_MSG)
    inline void
    get_option(T& x, std::string const& config)
    {
        if (!config.empty()) {
            x = boost::lexical_cast<T>(
                get_runtime().get_config().get_entry(config, x));
        }
    }
#endif

    typedef int (*hpx_main_type)(boost::program_options::variables_map&);

    /// \endcond

    /// \brief Main entry point for launching the HPX runtime system.
    ///
    /// This is the main entry point for any HPX application. This function
    /// (or one of its overloads below) should be called from the users `main()`
    /// function. It will set up the HPX runtime environment and schedule the
    /// function given by \p f as a HPX thread.
    ///
    /// \param f            [in] The function to be scheduled as an HPX
    ///                     thread. Usually this function represents the main
    ///                     entry point of any HPX application.
    /// \param desc_cmdline [in] This parameter may hold the description of
    ///                     additional command line arguments understood by the
    ///                     application. These options will be prepended to
    ///                     the default command line options understood by
    ///                     \a hpx::init (see description below).
    /// \param argc         [in] The number of command line arguments passed
    ///                     in \p argv. This is usually the unchanged value as
    ///                     passed by the operating system (to `main()`).
    /// \param argv         [in] The command line arguments for this
    ///                     application, usually that is the value as passed
    ///                     by the operating system (to `main()`).
    /// \param cfg          A list of configuration settings which will be added
    ///                     to the system configuration before the runtime
    ///                     instance is run. Each of the entries in this list
    ///                     must have the format of a fully defined key/value
    ///                     pair from an ini-file (for instance
    ///                     'hpx.component.enabled=1')
    /// \param startup      [in] A function to be executed inside a HPX
    ///                     thread before \p f is called. If this parameter
    ///                     is not given no function will be executed.
    /// \param shutdown     [in] A function to be executed inside an HPX
    ///                     thread while hpx::finalize is executed. If this
    ///                     parameter is not given no function will be
    ///                     executed.
    /// \param mode         [in] The mode the created runtime environment
    ///                     should be initialized in. There has to be exactly
    ///                     one locality in each HPX application which is
    ///                     executed in console mode (\a hpx::runtime_mode_console),
    ///                     all other localities have to be run in worker mode
    ///                     (\a hpx::runtime_mode_worker). Normally this is
    ///                     set up automatically, but sometimes it is necessary
    ///                     to explicitly specify the mode.
    ///
    /// \returns            The function returns the value, which has been
    ///                     returned from the user supplied \p f.
    ///
    /// \note               If the parameter \p mode is not given (defaulted),
    ///                     the created runtime system instance will be
    ///                     executed in console or worker mode depending on the
    ///                     command line arguments passed in `argc`/`argv`.
    ///                     Otherwise it will be executed as specified by the
    ///                     parameter\p mode.
    int init(int (*f)(boost::program_options::variables_map& vm),
        boost::program_options::options_description& desc_cmdline,
        int argc, char* argv[], std::vector<std::string> cfg,
        HPX_STD_FUNCTION<void()> startup = HPX_STD_FUNCTION<void()>(),
        HPX_STD_FUNCTION<void()> shutdown = HPX_STD_FUNCTION<void()>(),
        hpx::runtime_mode mode = hpx::runtime_mode_default);

    /// \brief Main entry point for launching the HPX runtime system.
    ///
    /// This is the main entry point for any HPX application. This function
    /// (or one of its overloads below) should be called from the users `main()`
    /// function. It will set up the HPX runtime environment and schedule the
    /// function given by \p f as a HPX thread.
    ///
    /// \param f            [in] The function to be scheduled as an HPX
    ///                     thread. Usually this function represents the main
    ///                     entry point of any HPX application.
    /// \param desc_cmdline [in] This parameter may hold the description of
    ///                     additional command line arguments understood by the
    ///                     application. These options will be prepended to
    ///                     the default command line options understood by
    ///                     \a hpx::init (see description below).
    /// \param argc         [in] The number of command line arguments passed
    ///                     in \p argv. This is usually the unchanged value as
    ///                     passed by the operating system (to `main()`).
    /// \param argv         [in] The command line arguments for this
    ///                     application, usually that is the value as passed
    ///                     by the operating system (to `main()`).
    /// \param startup      [in] A function to be executed inside a HPX
    ///                     thread before \p f is called. If this parameter
    ///                     is not given no function will be executed.
    /// \param shutdown     [in] A function to be executed inside an HPX
    ///                     thread while hpx::finalize is executed. If this
    ///                     parameter is not given no function will be
    ///                     executed.
    /// \param mode         [in] The mode the created runtime environment
    ///                     should be initialized in. There has to be exactly
    ///                     one locality in each HPX application which is
    ///                     executed in console mode (\a hpx::runtime_mode_console),
    ///                     all other localities have to be run in worker mode
    ///                     (\a hpx::runtime_mode_worker). Normally this is
    ///                     set up automatically, but sometimes it is necessary
    ///                     to explicitly specify the mode.
    ///
    /// \returns            The function returns the value, which has been
    ///                     returned from the user supplied \p f.
    ///
    /// \note               If the parameter \p mode is not given (defaulted),
    ///                     the created runtime system instance will be
    ///                     executed in console or worker mode depending on the
    ///                     command line arguments passed in `argc`/`argv`.
    ///                     Otherwise it will be executed as specified by the
    ///                     parameter\p mode.
    inline int
    init(int (*f)(boost::program_options::variables_map& vm),
        boost::program_options::options_description& desc_cmdline,
        int argc, char* argv[],
        HPX_STD_FUNCTION<void()> startup = HPX_STD_FUNCTION<void()>(),
        HPX_STD_FUNCTION<void()> shutdown = HPX_STD_FUNCTION<void()>(),
        hpx::runtime_mode mode = hpx::runtime_mode_default)
    {
        std::vector<std::string> cfg;
        return init(f, desc_cmdline, argc, argv, cfg, startup, shutdown, mode);
    }

    /// \fn int init(int (*f)(boost::program_options::variables_map& vm), std::string const& app_name, int argc, char* argv[])
    ///
    /// \brief Main entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main entry point, which can be used to set up the
    /// runtime for an HPX application (the runtime system will be set up in
    /// console mode or worker mode depending on the command line settings).
    ///
    /// \param f            [in] The function to be scheduled as an HPX
    ///                     thread. Usually this function represents the main
    ///                     entry point of any HPX application.
    /// \param app_name     [in] The name of the application.
    /// \param argc         [in] The number of command line arguments passed
    ///                     in \p argv. This is usually the unchanged value as
    ///                     passed by the operating system (to `main()`).
    /// \param argv         [in] The command line arguments for this
    ///                     application, usually that is the value as passed
    ///                     by the operating system (to `main()`).
    ///
    /// \returns            The function returns the value, which has been
    ///                     returned from the user supplied function \p f.
    ///
    /// \note               The created runtime system instance will be
    ///                     executed in console or worker mode depending on the
    ///                     command line arguments passed in `argc`/`argv`.
    int init(int (*f)(boost::program_options::variables_map& vm),
        std::string const& app_name, int argc, char* argv[]);

    /// \brief Main entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main entry point, which can be used to set up the
    /// runtime for an HPX application (the runtime system will be set up in
    /// console mode or worker mode depending on the command line settings).
    ///
    /// In console mode it will execute the user supplied function `hpx_main`,
    /// in worker mode it will execute an empty `hpx_main`.
    ///
    /// \param desc_cmdline [in] This parameter may hold the description of
    ///                     additional command line arguments understood by the
    ///                     application. These options will be prepended to
    ///                     the default command line options understood by
    ///                     \a hpx::init (see description below).
    /// \param argc         [in] The number of command line arguments passed
    ///                     in \p argv. This is usually the unchanged value as
    ///                     passed by the operating system (to `main()`).
    /// \param argv         [in] The command line arguments for this
    ///                     application, usually that is the value as passed
    ///                     by the operating system (to `main()`).
    /// \param startup      [in] A function to be executed inside a HPX
    ///                     thread before \p f is called. If this parameter
    ///                     is not given no function will be executed.
    /// \param shutdown     [in] A function to be executed inside an HPX
    ///                     thread while hpx::finalize is executed. If this
    ///                     parameter is not given no function will be
    ///                     executed.
    /// \param mode         [in] The mode the created runtime environment
    ///                     should be initialized in. There has to be exactly
    ///                     one locality in each HPX application which is
    ///                     executed in console mode (\a hpx::runtime_mode_console),
    ///                     all other localities have to be run in worker mode
    ///                     (\a hpx::runtime_mode_worker). Normally this is
    ///                     set up automatically, but sometimes it is necessary
    ///                     to explicitly specify the mode.
    ///
    /// \returns            The function returns the value, which has been
    ///                     returned from `hpx_main` (or 0 when executed in
    ///                     worker mode).
    ///
    /// \note               If the parameter \p mode is not given (defaulted),
    ///                     the created runtime system instance will be
    ///                     executed in console or worker mode depending on the
    ///                     command line arguments passed in `argc`/`argv`.
    ///                     Otherwise it will be executed as specified by the
    ///                     parameter\p mode.
    inline int
    init(boost::program_options::options_description& desc_cmdline,
        int argc, char* argv[],
        HPX_STD_FUNCTION<void()> startup = HPX_STD_FUNCTION<void()>(),
        HPX_STD_FUNCTION<void()> shutdown = HPX_STD_FUNCTION<void()>(),
        hpx::runtime_mode mode = hpx::runtime_mode_default)
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
    ///
    /// \param desc_cmdline [in] This parameter may hold the description of
    ///                     additional command line arguments understood by the
    ///                     application. These options will be prepended to
    ///                     the default command line options understood by
    ///                     \a hpx::init (see description below).
    /// \param argc         [in] The number of command line arguments passed
    ///                     in \p argv. This is usually the unchanged value as
    ///                     passed by the operating system (to `main()`).
    /// \param argv         [in] The command line arguments for this
    ///                     application, usually that is the value as passed
    ///                     by the operating system (to `main()`).
    /// \param cfg          A list of configuration settings which will be added
    ///                     to the system configuration before the runtime
    ///                     instance is run. Each of the entries in this list
    ///                     must have the format of a fully defined key/value
    ///                     pair from an ini-file (for instance
    ///                     'hpx.component.enabled=1')
    /// \param startup      [in] A function to be executed inside a HPX
    ///                     thread before \p f is called. If this parameter
    ///                     is not given no function will be executed.
    /// \param shutdown     [in] A function to be executed inside an HPX
    ///                     thread while hpx::finalize is executed. If this
    ///                     parameter is not given no function will be
    ///                     executed.
    /// \param mode         [in] The mode the created runtime environment
    ///                     should be initialized in. There has to be exactly
    ///                     one locality in each HPX application which is
    ///                     executed in console mode (\a hpx::runtime_mode_console),
    ///                     all other localities have to be run in worker mode
    ///                     (\a hpx::runtime_mode_worker). Normally this is
    ///                     set up automatically, but sometimes it is necessary
    ///                     to explicitly specify the mode.
    ///
    /// \returns            The function returns the value, which has been
    ///                     returned from `hpx_main` (or 0 when executed in
    ///                     worker mode).
    ///
    /// \note               If the parameter \p mode is not given (defaulted),
    ///                     the created runtime system instance will be
    ///                     executed in console or worker mode depending on the
    ///                     command line arguments passed in `argc`/`argv`.
    ///                     Otherwise it will be executed as specified by the
    ///                     parameter\p mode.
    inline int
    init(boost::program_options::options_description& desc_cmdline,
        int argc, char* argv[], std::vector<std::string> const& cfg,
        HPX_STD_FUNCTION<void()> startup = HPX_STD_FUNCTION<void()>(),
        HPX_STD_FUNCTION<void()> shutdown = HPX_STD_FUNCTION<void()>(),
        hpx::runtime_mode mode = hpx::runtime_mode_default)
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
    ///
    /// \param desc_cmdline [in] This parameter may hold the description of
    ///                     additional command line arguments understood by the
    ///                     application. These options will be prepended to
    ///                     the default command line options understood by
    ///                     \a hpx::init (see description below).
    /// \param argc         [in] The number of command line arguments passed
    ///                     in \p argv. This is usually the unchanged value as
    ///                     passed by the operating system (to `main()`).
    /// \param argv         [in] The command line arguments for this
    ///                     application, usually that is the value as passed
    ///                     by the operating system (to `main()`).
    /// \param mode         [in] The mode the created runtime environment
    ///                     should be initialized in. There has to be exactly
    ///                     one locality in each HPX application which is
    ///                     executed in console mode (\a hpx::runtime_mode_console),
    ///                     all other localities have to be run in worker mode
    ///                     (\a hpx::runtime_mode_worker). Normally this is
    ///                     set up automatically, but sometimes it is necessary
    ///                     to explicitly specify the mode.
    ///
    /// \returns            The function returns the value, which has been
    ///                     returned from `hpx_main` (or 0 when executed in
    ///                     worker mode).
    ///
    /// \note               If the parameter \p mode is runtime_mode_default,
    ///                     the created runtime system instance will be
    ///                     executed in console or worker mode depending on the
    ///                     command line arguments passed in `argc`/`argv`.
    ///                     Otherwise it will be executed as specified by the
    ///                     parameter\p mode.
    inline int
    init(boost::program_options::options_description& desc_cmdline, int argc,
        char* argv[], hpx::runtime_mode mode)
    {
        HPX_STD_FUNCTION<void()> const empty;
        return init(static_cast<hpx_main_type>(::hpx_main), desc_cmdline,
            argc, argv, empty, empty, mode);
    }

    /// \fn int init(std::string const& app_name, int argc = 0, char* argv[] = 0)
    ///
    /// \brief Main entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main entry point, which can be used to set up the
    /// runtime for an HPX application (the runtime system will be set up in
    /// console mode or worker mode depending on the command line settings).
    ///
    /// \param app_name     [in] The name of the application.
    /// \param argc         [in] The number of command line arguments passed
    ///                     in \p argv. This is usually the unchanged value as
    ///                     passed by the operating system (to `main()`).
    /// \param argv         [in] The command line arguments for this
    ///                     application, usually that is the value as passed
    ///                     by the operating system (to `main()`).
    ///
    /// \returns            The function returns the value, which has been
    ///                     returned from `hpx_main` (or 0 when executed in
    ///                     worker mode).
    ///
    /// \note               The created runtime system instance will be
    ///                     executed in console or worker mode depending on the
    ///                     command line arguments passed in `argc`/`argv`.
    inline int
    init(std::string const& app_name, int argc = 0, char* argv[] = 0)
    {
        return init(static_cast<hpx_main_type>(::hpx_main), app_name,
            argc, argv);
    }

    /// \brief Main entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main entry point, which can be used to set up the
    /// runtime for an HPX application (the runtime system will be set up in
    /// console mode or worker mode depending on the command line settings).
    ///
    /// \param argc         [in] The number of command line arguments passed
    ///                     in \p argv. This is usually the unchanged value as
    ///                     passed by the operating system (to `main()`).
    /// \param argv         [in] The command line arguments for this
    ///                     application, usually that is the value as passed
    ///                     by the operating system (to `main()`).
    ///
    /// \returns            The function returns the value, which has been
    ///                     returned from `hpx_main` (or 0 when executed in
    ///                     worker mode).
    ///
    /// \note               The created runtime system instance will be
    ///                     executed in console or worker mode depending on the
    ///                     command line arguments passed in `argc`/`argv`. If not
    ///                     command line arguments are passed, console mode is
    ///                     assumed.
    ///
    /// \note               If no command line arguments are passed the HPX
    ///                     runtime system will not support any of the default
    ///                     command line options as described in the section
    ///                     'HPX Command Line Options'.
    inline int
    init(int argc = 0, char* argv[] = 0)
    {
        return init(static_cast<hpx_main_type>(::hpx_main),
            HPX_APPLICATION_STRING, argc, argv);
    }

    /// \brief Main entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main entry point, which can be used to set up the
    /// runtime for an HPX application (the runtime system will be set up in
    /// console mode or worker mode depending on the command line settings).
    ///
    /// \param f            [in] The function to be scheduled as an HPX
    ///                     thread. Usually this function represents the main
    ///                     entry point of any HPX application.
    /// \param app_name     [in] The name of the application.
    /// \param argc         [in] The number of command line arguments passed
    ///                     in \p argv. This is usually the unchanged value as
    ///                     passed by the operating system (to `main()`).
    /// \param argv         [in] The command line arguments for this
    ///                     application, usually that is the value as passed
    ///                     by the operating system (to `main()`).
    ///
    /// \returns            The function returns the value, which has been
    ///                     returned from the user supplied function \p f.
    ///
    /// \note               The created runtime system instance will be
    ///                     executed in console or worker mode depending on the
    ///                     command line arguments passed in `argc`/`argv`.
    inline int init(int (*f)(boost::program_options::variables_map& vm),
        std::string const& app_name, int argc, char* argv[])
    {
        using boost::program_options::options_description;

        options_description desc_commandline(
            "Usage: " + app_name +  " [options]");

        if (argc == 0 || argv == 0)
        {
            char *dummy_argv[1] = { const_cast<char*>(app_name.c_str()) };
            return init(desc_commandline, 1, dummy_argv);
        }

        HPX_STD_FUNCTION<void()> const empty;
        return init(f, desc_commandline, argc, argv, empty, empty);
    }

    /// \brief Main function to gracefully terminate the the HPX runtime system.
    ///
    /// The function \a hpx::finalize is the main way to (gracefully) exit any
    /// HPX application. It should be called from one locality only (usually
    /// the console) and it will notify all connected localities to finish
    /// execution. Only after all other localities have exited this function
    /// will return, allowing to exit the console locality as well.
    ///
    /// During the execution of this function the runtime system will invoke
    /// all registered shutdown functions (see \a hpx::init) on all localities.
    ///
    /// \param shutdown_timeout This parameter allows to specify a timeout (in
    ///           microseconds), specifying how long any of the connected
    ///           localities should wait for pending tasks to be executed.
    ///           After this timeout, all suspended HPX-threads will be aborted.
    ///           Note, that this function will not abort any running
    ///           HPX-threads. In any case the shutdown will not proceed as long
    ///           as there is at least one pending/running HPX-thread.
    ///
    ///           The default value (`-1.0`) will try to find a globally set
    ///           timeout value (can be set as the configuration parameter
    ///           `hpx.shutdown_timeout`), and if that is not set or `-1.0` as
    ///           well, it will disable any timeout, each connected
    ///           locality will wait for all existing HPX-threads to terminate.
    ///
    /// \param localwait This parameter allows to specify a local wait time
    ///           (in microseconds) before the connected localities will be
    ///           notified and the overall shutdown process starts.
    ///
    ///           The default value (`-1.0`) will try to find a globally set
    ///           wait time value (can be set as the configuration parameter
    ///           "hpx.finalize_wait_time"), and if this is not set or `-1.0`
    ///           as well, it will disable any addition local wait time before
    ///           proceeding.
    ///
    /// This function will block and wait for all connected localities to exit
    /// before returning to the caller. It should be the last HPX-function
    /// called by any application.
    ///
    /// Using this function is an alternative to \a hpx::disconnect, these
    /// functions do not need to be called both.
    int finalize(double shutdown_timeout = -1.0,
        double localwait = -1.0);

    /// \brief Terminate any application non-gracefully.
    ///
    /// The function \a hpx::terminate is the non-graceful way to exit any
    /// application immediately. It can be called from any locality and will
    /// terminate all localities currently used by the application.
    ///
    /// \note    This function will cause HPX to call `std::terminate()` on
    ///          all localities associated with this application.
    void terminate();

    /// \brief Disconnect this locality from the application.
    ///
    /// The function \a hpx::disconnect can be used to disconnect a locality
    /// from a running HPX application.
    ///
    /// During the execution of this function the runtime system will invoke
    /// all registered shutdown functions (see \a hpx::init) on this locality.
    //
    /// \param shutdown_timeout This parameter allows to specify a timeout (in
    ///           microseconds), specifying how long this locality should wait
    ///           for pending tasks to be executed. After this timeout, all
    ///           suspended HPX-threads will be aborted.
    ///           Note, that this function will not abort any running
    ///           HPX-threads. In any case the shutdown will not proceed as long
    ///           as there is at least one pending/running HPX-thread.
    ///
    ///           The default value (`-1.0`) will try to find a globally set
    ///           timeout value (can be set as the configuration parameter
    ///           "hpx.shutdown_timeout"), and if that is not set or `-1.0` as
    ///           well, it will disable any timeout, each connected
    ///           locality will wait for all existing HPX-threads to terminate.
    ///
    /// \param localwait This parameter allows to specify a local wait time
    ///           (in microseconds) before the connected localities will be
    ///           notified and the overall shutdown process starts.
    ///
    ///           The default value (`-1.0`) will try to find a globally set
    ///           wait time value (can be set as the configuration parameter
    ///           `hpx.finalize_wait_time`), and if this is not set or `-1.0`
    ///           as well, it will disable any addition local wait time before
    ///           proceeding.
    ///
    /// This function will block and wait for this locality to finish executing
    /// before returning to the caller. It should be the last HPX-function
    /// called by any locality being disconnected.
    void disconnect(double shutdown_timeout = -1.0,
        double localwait = -1.0);
}

#endif // HPX_ABC9B037_3A25_4591_BB60_CD166773D61D

