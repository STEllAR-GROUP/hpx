//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx_start.hpp

#if !defined(HPX_START_OCT_04_2012_0148PM)
#define HPX_START_OCT_04_2012_0148PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/hpx_finalize.hpp>
#include <hpx/hpx_user_main_config.hpp>

#include <boost/program_options.hpp>

#include <string>

#ifndef DOXYGEN
///////////////////////////////////////////////////////////////////////////////
// One of these functions must be implemented by the application for the
// console locality.
int hpx_main();
int hpx_main(int argc, char** argv);
int hpx_main(boost::program_options::variables_map& vm);
#endif

///////////////////////////////////////////////////////////////////////////////
/// \namespace hpx_startup
namespace hpx_startup
{
#ifndef DOXYGEN
    // As an alternative, the user can provide a function hpx_startup::user_main,
    // which is semantically equivalent to the plain old C-main.
    int user_main();
    int user_main(int argc, char** argv);
#endif
}

///////////////////////////////////////////////////////////////////////////////
/// \namespace hpx
namespace hpx
{
#ifndef DOXYGEN
    typedef int (*hpx_main_type)(boost::program_options::variables_map&);
#endif

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is the main, non-blocking entry point for any HPX application.
    /// This function (or one of its overloads below) should be called from
    /// the users `main()` function. It will set up the HPX runtime environment
    /// and schedule the function given by \p f as a HPX thread. It will return
    /// immediatly after that. Use `hpx::wait` and `hpx::stop` to synchronize
    /// with the runtime system's execution.
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
    /// \returns            The function returns \a true if command line processing
    ///                     succeeded and the runtime system was started successfully.
    ///                     It will return \a false otherwise.
    ///
    /// \note               If the parameter \p mode is not given (defaulted),
    ///                     the created runtime system instance will be
    ///                     executed in console or worker mode depending on the
    ///                     command line arguments passed in `argc`/`argv`.
    ///                     Otherwise it will be executed as specified by the
    ///                     parameter\p mode.
    inline bool start(
        util::function_nonser<int(boost::program_options::variables_map& vm)> const& f,
        boost::program_options::options_description const& desc_cmdline,
        int argc, char** argv, std::vector<std::string> const& cfg,
        util::function_nonser<void()> const& startup = util::function_nonser<void()>(),
        util::function_nonser<void()> const& shutdown = util::function_nonser<void()>(),
        hpx::runtime_mode mode = hpx::runtime_mode_default);

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is the main, non-blocking entry point for any HPX application.
    /// This function (or one of its overloads below) should be called from the
    /// users `main()` function. It will set up the HPX runtime environment and
    /// schedule the function given by \p f as a HPX thread. It will return
    /// immediatly after that. Use `hpx::wait` and `hpx::stop` to synchronize
    /// with the runtime system's execution.
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
    /// \returns            The function returns \a true if command line processing
    ///                     succeeded and the runtime system was started successfully.
    ///                     It will return \a false otherwise.
    ///
    /// \note               If the parameter \p mode is not given (defaulted),
    ///                     the created runtime system instance will be
    ///                     executed in console or worker mode depending on the
    ///                     command line arguments passed in `argc`/`argv`.
    ///                     Otherwise it will be executed as specified by the
    ///                     parameter\p mode.
    inline bool
    start(int (*f)(boost::program_options::variables_map& vm),
        boost::program_options::options_description const& desc_cmdline,
        int argc, char** argv,
        util::function_nonser<void()> const& startup = util::function_nonser<void()>(),
        util::function_nonser<void()> const& shutdown = util::function_nonser<void()>(),
        hpx::runtime_mode mode = hpx::runtime_mode_default);

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main, non-blocking entry point, which can be used
    /// to set up the runtime for an HPX application (the runtime system will be
    /// set up in console mode or worker mode depending on the command line
    /// settings). It will return immediatly after that. Use `hpx::wait` and
    /// `hpx::stop` to synchronize with the runtime system's execution.
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
    /// \returns            The function returns \a true if command line processing
    ///                     succeeded and the runtime system was started successfully.
    ///                     It will return \a false otherwise.
    ///
    /// \note               If the parameter \p mode is not given (defaulted),
    ///                     the created runtime system instance will be
    ///                     executed in console or worker mode depending on the
    ///                     command line arguments passed in `argc`/`argv`.
    ///                     Otherwise it will be executed as specified by the
    ///                     parameter\p mode.
    inline bool
    start(boost::program_options::options_description const& desc_cmdline,
        int argc, char** argv,
        util::function_nonser<void()> const& startup = util::function_nonser<void()>(),
        util::function_nonser<void()> const& shutdown = util::function_nonser<void()>(),
        hpx::runtime_mode mode = hpx::runtime_mode_default);

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main, non-blocking entry point, which can be used
    /// to set up the runtime for an HPX application (the runtime system will
    /// be set up in console mode or worker mode depending on the command line
    /// settings). It will return immediatly after that. Use `hpx::wait` and
    /// `hpx::stop` to synchronize with the runtime system's execution.
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
    /// \returns            The function returns \a true if command line processing
    ///                     succeeded and the runtime system was started successfully.
    ///                     It will return \a false otherwise.
    ///
    /// \note               If the parameter \p mode is not given (defaulted),
    ///                     the created runtime system instance will be
    ///                     executed in console or worker mode depending on the
    ///                     command line arguments passed in `argc`/`argv`.
    ///                     Otherwise it will be executed as specified by the
    ///                     parameter\p mode.
    inline bool
    start(boost::program_options::options_description const& desc_cmdline,
        int argc, char** argv, std::vector<std::string> const& cfg,
        util::function_nonser<void()> const& startup = util::function_nonser<void()>(),
        util::function_nonser<void()> const& shutdown = util::function_nonser<void()>(),
        hpx::runtime_mode mode = hpx::runtime_mode_default);

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main, non-blocking entry point, which can be used
    /// to set up the runtime for an HPX application (the runtime system will
    /// be set up in console mode or worker mode depending on the command line
    /// settings). It will return immediatly after that. Use `hpx::wait` and
    /// `hpx::stop` to synchronize with the runtime system's execution.
    ///
    /// In console mode it will execute the user supplied function `hpx_main`,
    /// in worker mode it will execute an empty `hpx_main`.
    ///
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
    /// \param mode         [in] The mode the created runtime environment
    ///                     should be initialized in. There has to be exactly
    ///                     one locality in each HPX application which is
    ///                     executed in console mode (\a hpx::runtime_mode_console),
    ///                     all other localities have to be run in worker mode
    ///                     (\a hpx::runtime_mode_worker). Normally this is
    ///                     set up automatically, but sometimes it is necessary
    ///                     to explicitly specify the mode.
    ///
    /// \returns            The function returns \a true if command line processing
    ///                     succeeded and the runtime system was started successfully.
    ///                     It will return \a false otherwise.
    ///
    /// \note               If the parameter \p mode is runtime_mode_default,
    ///                     the created runtime system instance will be
    ///                     executed in console or worker mode depending on the
    ///                     command line arguments passed in `argc`/`argv`.
    ///                     Otherwise it will be executed as specified by the
    ///                     parameter\p mode.
    inline bool
    start(int argc, char** argv, std::vector<std::string> const& cfg,
        hpx::runtime_mode mode = hpx::runtime_mode_default);

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main, non-blocking entry point, which can be used
    /// to set up the runtime for an HPX application (the runtime system will
    /// be set up in console mode or worker mode depending on the command line
    /// settings). It will return immediatly after that. Use `hpx::wait` and
    /// `hpx::stop` to synchronize with the runtime system's execution.
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
    /// \returns            The function returns \a true if command line processing
    ///                     succeeded and the runtime system was started successfully.
    ///                     It will return \a false otherwise.
    ///
    /// \note               If the parameter \p mode is runtime_mode_default,
    ///                     the created runtime system instance will be
    ///                     executed in console or worker mode depending on the
    ///                     command line arguments passed in `argc`/`argv`.
    ///                     Otherwise it will be executed as specified by the
    ///                     parameter\p mode.
    inline bool
    start(boost::program_options::options_description const& desc_cmdline,
        int argc, char** argv, hpx::runtime_mode mode);

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main, non-blocking entry point, which can be used
    /// to set up the runtime for an HPX application (the runtime system will
    /// be set up in console mode or worker mode depending on the command line
    /// settings). It will return immediatly after that. Use `hpx::wait` and
    /// `hpx::stop` to synchronize with the runtime system's execution.
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
    /// \returns            The function returns \a true if command line processing
    ///                     succeeded and the runtime system was started successfully.
    ///                     It will return \a false otherwise.
    ///
    /// \note               If the parameter \p mode is runtime_mode_default,
    ///                     the created runtime system instance will be
    ///                     executed in console or worker mode depending on the
    ///                     command line arguments passed in `argc`/`argv`.
    ///                     Otherwise it will be executed as specified by the
    ///                     parameter\p mode.
    inline bool
    start(boost::program_options::options_description const& desc_cmdline,
        int argc, char** argv, std::vector<std::string> const& cfg,
        hpx::runtime_mode mode);

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main, non-blocking entry point, which can be used
    /// to set up the runtime for an HPX application (the runtime system will
    /// be set up in console mode or worker mode depending on the command line
    /// settings). It will return immediatly after that. Use `hpx::wait` and
    /// `hpx::stop` to synchronize with the runtime system's execution.
    ///
    /// \param app_name     [in] The name of the application.
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
    /// \returns            The function returns \a true if command line processing
    ///                     succeeded and the runtime system was started successfully.
    ///                     It will return \a false otherwise.
    ///
    /// \note               The created runtime system instance will be
    ///                     executed in console or worker mode depending on the
    ///                     command line arguments passed in `argc`/`argv`.
    inline bool
    start(std::string const& app_name, int argc = 0, char** argv = 0,
        hpx::runtime_mode mode = hpx::runtime_mode_default);

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main, non-blocking entry point, which can be used
    /// to set up the runtime for an HPX application (the runtime system will
    /// be set up in console mode or worker mode depending on the command line
    /// settings). It will return immediatly after that. Use `hpx::wait` and
    /// `hpx::stop` to synchronize with the runtime system's execution.
    ///
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
    /// \returns            The function returns \a true if command line processing
    ///                     succeeded and the runtime system was started successfully.
    ///                     It will return \a false otherwise.
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
    inline bool start(int argc = 0, char** argv = 0,
        hpx::runtime_mode mode = hpx::runtime_mode_default);

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main, non-blocking entry point, which can be used
    /// to set up the runtime for an HPX application (the runtime system will
    /// be set up in console mode or worker mode depending on the command line
    /// settings). It will return immediatly after that. Use `hpx::wait` and
    /// `hpx::stop` to synchronize with the runtime system's execution.
    ///
    /// \param cfg          A list of configuration settings which will be added
    ///                     to the system configuration before the runtime
    ///                     instance is run. Each of the entries in this list
    ///                     must have the format of a fully defined key/value
    ///                     pair from an ini-file (for instance
    ///                     'hpx.component.enabled=1')
    /// \param mode         [in] The mode the created runtime environment
    ///                     should be initialized in. There has to be exactly
    ///                     one locality in each HPX application which is
    ///                     executed in console mode (\a hpx::runtime_mode_console),
    ///                     all other localities have to be run in worker mode
    ///                     (\a hpx::runtime_mode_worker). Normally this is
    ///                     set up automatically, but sometimes it is necessary
    ///                     to explicitly specify the mode.
    ///
    /// \returns            The function returns \a true if command line processing
    ///                     succeeded and the runtime system was started successfully.
    ///                     It will return \a false otherwise.
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
    inline bool start(std::vector<std::string> const& cfg,
        hpx::runtime_mode mode = hpx::runtime_mode_default);

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main, non-blocking entry point, which can be used
    /// to set up the runtime for an HPX application (the runtime system will
    /// be set up in console mode or worker mode depending on the command line
    /// settings). It will return immediatly after that. Use `hpx::wait` and
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
    /// \param mode         [in] The mode the created runtime environment
    ///                     should be initialized in. There has to be exactly
    ///                     one locality in each HPX application which is
    ///                     executed in console mode (\a hpx::runtime_mode_console),
    ///                     all other localities have to be run in worker mode
    ///                     (\a hpx::runtime_mode_worker). Normally this is
    ///                     set up automatically, but sometimes it is necessary
    ///                     to explicitly specify the mode.
    ///
    /// \returns            The function returns \a true if command line processing
    ///                     succeeded and the runtime system was started successfully.
    ///                     It will return \a false otherwise.
    ///
    /// \note               The created runtime system instance will be
    ///                     executed in console or worker mode depending on the
    ///                     command line arguments passed in `argc`/`argv`.
    inline bool start(int (*f)(boost::program_options::variables_map& vm),
        std::string const& app_name, int argc, char** argv,
        hpx::runtime_mode mode = hpx::runtime_mode_default);

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main, non-blocking entry point, which can be used
    /// to set up the runtime for an HPX application (the runtime system will
    /// be set up in console mode or worker mode depending on the command line
    /// settings). It will return immediatly after that. Use `hpx::wait` and
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
    /// \param mode         [in] The mode the created runtime environment
    ///                     should be initialized in. There has to be exactly
    ///                     one locality in each HPX application which is
    ///                     executed in console mode (\a hpx::runtime_mode_console),
    ///                     all other localities have to be run in worker mode
    ///                     (\a hpx::runtime_mode_worker). Normally this is
    ///                     set up automatically, but sometimes it is necessary
    ///                     to explicitly specify the mode.
    ///
    /// \returns            The function returns \a true if command line processing
    ///                     succeeded and the runtime system was started successfully.
    ///                     It will return \a false otherwise.
    ///
    /// \note               The created runtime system instance will be
    ///                     executed in console or worker mode depending on the
    ///                     command line arguments passed in `argc`/`argv`.
    inline bool start(util::function_nonser<int(int, char**)> const&,
        std::string const& app_name, int argc, char** argv,
        hpx::runtime_mode mode = hpx::runtime_mode_default);

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main, non-blocking entry point, which can be used
    /// to set up the runtime for an HPX application (the runtime system will
    /// be set up in console mode or worker mode depending on the command line
    /// settings). It will return immediatly after that. Use `hpx::wait` and
    ///
    /// \param f            [in] The function to be scheduled as an HPX
    ///                     thread. Usually this function represents the main
    ///                     entry point of any HPX application.
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
    /// \returns            The function returns \a true if command line processing
    ///                     succeeded and the runtime system was started successfully.
    ///                     It will return \a false otherwise.
    ///
    /// \note               The created runtime system instance will be
    ///                     executed in console or worker mode depending on the
    ///                     command line arguments passed in `argc`/`argv`.
    inline bool start(int (*f)(boost::program_options::variables_map& vm),
        int argc, char** argv, hpx::runtime_mode mode = hpx::runtime_mode_default);

    /// \brief Main non-blocking entry point for launching the HPX runtime system.
    ///
    /// This is a simplified main, non-blocking entry point, which can be used
    /// to set up the runtime for an HPX application (the runtime system will
    /// be set up in console mode or worker mode depending on the command line
    /// settings). It will return immediatly after that. Use `hpx::wait` and
    ///
    /// \param f            [in] The function to be scheduled as an HPX
    ///                     thread. Usually this function represents the main
    ///                     entry point of any HPX application.
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
    /// \returns            The function returns \a true if command line processing
    ///                     succeeded and the runtime system was started successfully.
    ///                     It will return \a false otherwise.
    ///
    /// \note               The created runtime system instance will be
    ///                     executed in console or worker mode depending on the
    ///                     command line arguments passed in `argc`/`argv`.
    inline bool start(util::function_nonser<int(int, char**)> const& f,
        int argc, char** argv, hpx::runtime_mode mode = hpx::runtime_mode_default);
}

#ifndef DOXYGEN
///////////////////////////////////////////////////////////////////////////////
// Pull in the implementation of the inlined hpx::init functions if we're not
// compiling the core HPX library.
#if !defined(HPX_EXPORTS)
#  include <hpx/hpx_start_impl.hpp>
#endif
#endif

#endif

