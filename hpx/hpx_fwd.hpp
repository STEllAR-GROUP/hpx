//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx_fwd.hpp

#if !defined(HPX_HPX_FWD_MAR_24_2008_1119AM)
#define HPX_HPX_FWD_MAR_24_2008_1119AM

#include <hpx/config.hpp>

#include <cstdlib>
#include <vector>

#include <boost/config.hpp>
#include <boost/version.hpp>

#if BOOST_VERSION < 104900
// Please update your Boost installation (see www.boost.org for details).
#error HPX cannot be compiled with a Boost version earlier than V1.49.
#endif

#if defined(HPX_WINDOWS)
#if !defined(WIN32)
#  define WIN32
#endif
#include <winsock2.h>
#include <windows.h>
#endif

#include <boost/shared_ptr.hpp>
#include <boost/intrusive_ptr.hpp>
#include <boost/cstdint.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/system/error_code.hpp>
#include <boost/type_traits/remove_reference.hpp>

#include <hpx/traits.hpp>
#include <hpx/exception_fwd.hpp>
#include <hpx/traits/component_type_database.hpp>
#include <hpx/lcos/local/once_fwd.hpp>
#include <hpx/lcos_fwd.hpp>
#include <hpx/runtime_fwd.hpp>
#include <hpx/util/function.hpp>
// ^ this has to come before the naming/id_type.hpp below
#include <hpx/util/move.hpp>
#include <hpx/util/unique_function.hpp>
#include <hpx/util/unused.hpp>
#include <hpx/runtime/agas_fwd.hpp>
#include <hpx/runtime/applier_fwd.hpp>
#include <hpx/runtime/components_fwd.hpp>
#include <hpx/runtime/find_here.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/naming_fwd.hpp>
#include <hpx/runtime/parcelset_fwd.hpp>
#include <hpx/runtime/runtime_mode.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/threads_fwd.hpp>
#include <hpx/runtime/threads/detail/tagged_thread_state.hpp>
#include <hpx/runtime/threads/thread_enums.hpp>

/// \namespace hpx
///
/// The namespace \a hpx is the main namespace of the HPX library. All classes
/// functions and variables are defined inside this namespace.
namespace hpx
{
    /// \cond NOINTERNAL
    class error_code;

    HPX_EXCEPTION_EXPORT extern error_code throws;

    /// \namespace util
    namespace util
    {
        struct binary_filter;

        class HPX_EXPORT section;

        /// \brief Expand INI variables in a string
        HPX_API_EXPORT std::string expand(std::string const& expand);

        /// \brief Expand INI variables in a string
        HPX_API_EXPORT void expand(std::string& expand);
    }

    namespace performance_counters
    {
        struct counter_info;
    }

    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    // Pulling important types into the main namespace
    using naming::invalid_id;
}

namespace hpx
{

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the global id representing the root locality
    ///
    /// The function \a find_root_locality() can be used to retrieve the global
    /// id usable to refer to the root locality. The root locality is the
    /// locality where the main AGAS service is hosted.
    ///
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \note     Generally, the id of a locality can be used for instance to
    ///           create new instances of components and to invoke plain actions
    ///           (global functions).
    ///
    /// \returns  The global id representing the root locality for this
    ///           application.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// \note     This function will return meaningful results only if called
    ///           from an HPX-thread. It will return \a hpx::naming::invalid_id
    ///           otherwise.
    ///
    /// \see      \a hpx::find_all_localities(), \a hpx::find_locality()
    HPX_API_EXPORT naming::id_type find_root_locality(error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the list of global ids representing all localities
    ///        available to this application.
    ///
    /// The function \a find_all_localities() can be used to retrieve the
    /// global ids of all localities currently available to this application.
    ///
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \note     Generally, the id of a locality can be used for instance to
    ///           create new instances of components and to invoke plain actions
    ///           (global functions).
    ///
    /// \returns  The global ids representing the localities currently
    ///           available to this application.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// \note     This function will return meaningful results only if called
    ///           from an HPX-thread. It will return an empty vector otherwise.
    ///
    /// \see      \a hpx::find_here(), \a hpx::find_locality()
    HPX_API_EXPORT std::vector<naming::id_type> find_all_localities(
        error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the list of global ids representing all localities
    ///        available to this application which support the given component
    ///        type.
    ///
    /// The function \a find_all_localities() can be used to retrieve the
    /// global ids of all localities currently available to this application
    /// which support the creation of instances of the given component type.
    ///
    /// \note     Generally, the id of a locality can be used for instance to
    ///           create new instances of components and to invoke plain actions
    ///           (global functions).
    ///
    /// \param type  [in] The type of the components for which the function should
    ///           return the available localities.
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \returns  The global ids representing the localities currently
    ///           available to this application which support the creation of
    ///           instances of the given component type. If no localities
    ///           supporting the given component type are currently available,
    ///           this function will return an empty vector.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// \note     This function will return meaningful results only if called
    ///           from an HPX-thread. It will return an empty vector otherwise.
    ///
    /// \see      \a hpx::find_here(), \a hpx::find_locality()
    HPX_API_EXPORT std::vector<naming::id_type> find_all_localities(
        components::component_type type, error_code& ec = throws);

    /// \brief Return the list of locality ids of remote localities supporting
    ///        the given component type. By default this function will return
    ///        the list of all remote localities (all but the current locality).
    ///
    /// The function \a find_remote_localities() can be used to retrieve the
    /// global ids of all remote localities currently available to this
    /// application (i.e. all localities except the current one).
    ///
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \note     Generally, the id of a locality can be used for instance to
    ///           create new instances of components and to invoke plain actions
    ///           (global functions).
    ///
    /// \returns  The global ids representing the remote localities currently
    ///           available to this application.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// \note     This function will return meaningful results only if called
    ///           from an HPX-thread. It will return an empty vector otherwise.
    ///
    /// \see      \a hpx::find_here(), \a hpx::find_locality()
    HPX_API_EXPORT std::vector<naming::id_type> find_remote_localities(
        error_code& ec = throws);

    /// \brief Return the list of locality ids of remote localities supporting
    ///        the given component type. By default this function will return
    ///        the list of all remote localities (all but the current locality).
    ///
    /// The function \a find_remote_localities() can be used to retrieve the
    /// global ids of all remote localities currently available to this
    /// application (i.e. all localities except the current one) which
    /// support the creation of instances of the given component type.
    ///
    /// \param type  [in] The type of the components for which the function should
    ///           return the available remote localities.
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \note     Generally, the id of a locality can be used for instance to
    ///           create new instances of components and to invoke plain actions
    ///           (global functions).
    ///
    /// \returns  The global ids representing the remote localities currently
    ///           available to this application.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// \note     This function will return meaningful results only if called
    ///           from an HPX-thread. It will return an empty vector otherwise.
    ///
    /// \see      \a hpx::find_here(), \a hpx::find_locality()
    HPX_API_EXPORT std::vector<naming::id_type> find_remote_localities(
        components::component_type type, error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the global id representing an arbitrary locality which
    ///        supports the given component type.
    ///
    /// The function \a find_locality() can be used to retrieve the
    /// global id of an arbitrary locality currently available to this
    /// application which supports the creation of instances of the given
    /// component type.
    ///
    /// \note     Generally, the id of a locality can be used for instance to
    ///           create new instances of components and to invoke plain actions
    ///           (global functions).
    ///
    /// \param type  [in] The type of the components for which the function should
    ///           return any available locality.
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \returns  The global id representing an arbitrary locality currently
    ///           available to this application which supports the creation of
    ///           instances of the given component type. If no locality
    ///           supporting the given component type is currently available,
    ///           this function will return \a hpx::naming::invalid_id.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// \note     This function will return meaningful results only if called
    ///           from an HPX-thread. It will return \a hpx::naming::invalid_id
    ///           otherwise.
    ///
    /// \see      \a hpx::find_here(), \a hpx::find_all_localities()
    HPX_API_EXPORT naming::id_type find_locality(components::component_type type,
        error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the number of localities which are currently registered
    ///        for the running application.
    ///
    /// The function \a get_num_localities returns the number of localities
    /// currently connected to the console.
    ///
    /// \note     This function will return meaningful results only if called
    ///           from an HPX-thread. It will return 0 otherwise.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// \see      \a hpx::find_all_localities_sync, \a hpx::get_num_localities
    HPX_API_EXPORT boost::uint32_t get_num_localities_sync(error_code& ec = throws);

    /// \brief Return the number of localities which were registered at startup
    ///        for the running application.
    ///
    /// The function \a get_initial_num_localities returns the number of localities
    /// which were connected to the console at application startup.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// \see      \a hpx::find_all_localities, \a hpx::get_num_localities
    HPX_API_EXPORT boost::uint32_t get_initial_num_localities();

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Asynchronously return the number of localities which are
    ///        currently registered for the running application.
    ///
    /// The function \a get_num_localities asynchronously returns the
    /// number of localities currently connected to the console. The returned
    /// future represents the actual result.
    ///
    /// \note     This function will return meaningful results only if called
    ///           from an HPX-thread. It will return 0 otherwise.
    ///
    /// \see      \a hpx::find_all_localities, \a hpx::get_num_localities
    HPX_API_EXPORT lcos::future<boost::uint32_t> get_num_localities();

    /// \brief Return the number of localities which are currently registered
    ///        for the running application.
    ///
    /// The function \a get_num_localities returns the number of localities
    /// currently connected to the console which support the creation of the
    /// given component type.
    ///
    /// \param t  The component type for which the number of connected
    ///           localities should be retrieved.
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \note     This function will return meaningful results only if called
    ///           from an HPX-thread. It will return 0 otherwise.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// \see      \a hpx::find_all_localities, \a hpx::get_num_localities
    HPX_API_EXPORT boost::uint32_t get_num_localities_sync(
        components::component_type t, error_code& ec = throws);

    /// \brief Asynchronously return the number of localities which are
    ///        currently registered for the running application.
    ///
    /// The function \a get_num_localities asynchronously returns the
    /// number of localities currently connected to the console which support
    /// the creation of the given component type. The returned future represents
    /// the actual result.
    ///
    /// \param t  The component type for which the number of connected
    ///           localities should be retrieved.
    ///
    /// \note     This function will return meaningful results only if called
    ///           from an HPX-thread. It will return 0 otherwise.
    ///
    /// \see      \a hpx::find_all_localities, \a hpx::get_num_localities
    HPX_API_EXPORT lcos::future<boost::uint32_t> get_num_localities(
        components::component_type t);

    ///////////////////////////////////////////////////////////////////////////
    /// The type of a function which is registered to be executed as a
    /// startup or pre-startup function.
    typedef util::function_nonser<void()> startup_function_type;

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Add a function to be executed by a HPX thread before hpx_main
    /// but guaranteed before any startup function is executed (system-wide).
    ///
    /// Any of the functions registered with \a register_pre_startup_function
    /// are guaranteed to be executed by an HPX thread before any of the
    /// registered startup functions are executed (see
    /// \a hpx::register_startup_function()).
    ///
    /// \param f  [in] The function to be registered to run by an HPX thread as
    ///           a pre-startup function.
    ///
    /// \note If this function is called while the pre-startup functions are
    ///       being executed or after that point, it will raise a invalid_status
    ///       exception.
    ///
    ///       This function is one of the few API functions which can be called
    ///       before the runtime system has been fully initialized. It will
    ///       automatically stage the provided startup function to the runtime
    ///       system during its initialization (if necessary).
    ///
    /// \see    \a hpx::register_startup_function()
    HPX_API_EXPORT void register_pre_startup_function(startup_function_type const& f);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Add a function to be executed by a HPX thread before hpx_main
    /// but guaranteed after any pre-startup function is executed (system-wide).
    ///
    /// Any of the functions registered with \a register_startup_function
    /// are guaranteed to be executed by an HPX thread after any of the
    /// registered pre-startup functions are executed (see:
    /// \a hpx::register_pre_startup_function()), but before \a hpx_main is
    /// being called.
    ///
    /// \param f  [in] The function to be registered to run by an HPX thread as
    ///           a startup function.
    ///
    /// \note If this function is called while the startup functions are
    ///       being executed or after that point, it will raise a invalid_status
    ///       exception.
    ///
    ///       This function is one of the few API functions which can be called
    ///       before the runtime system has been fully initialized. It will
    ///       automatically stage the provided startup function to the runtime
    ///       system during its initialization (if necessary).
    ///
    /// \see    \a hpx::register_pre_startup_function()
    HPX_API_EXPORT void register_startup_function(startup_function_type const& f);

    /// The type of a function which is registered to be executed as a
    /// shutdown or pre-shutdown function.
    typedef util::function_nonser<void()> shutdown_function_type;

    /// \brief Add a function to be executed by a HPX thread during
    /// \a hpx::finalize() but guaranteed before any shutdown function is
    /// executed (system-wide)
    ///
    /// Any of the functions registered with \a register_pre_shutdown_function
    /// are guaranteed to be executed by an HPX thread during the execution of
    /// \a hpx::finalize() before any of the registered shutdown functions are
    /// executed (see: \a hpx::register_shutdown_function()).
    ///
    /// \param f  [in] The function to be registered to run by an HPX thread as
    ///           a pre-shutdown function.
    ///
    /// \note If this function is called while the pre-shutdown functions are
    ///       being executed, or after that point, it will raise a invalid_status
    ///       exception.
    ///
    /// \see    \a hpx::register_shutdown_function()
    HPX_API_EXPORT void register_pre_shutdown_function(shutdown_function_type const& f);

    /// \brief Add a function to be executed by a HPX thread during
    /// \a hpx::finalize() but guaranteed after any pre-shutdown function is
    /// executed (system-wide)
    ///
    /// Any of the functions registered with \a register_shutdown_function
    /// are guaranteed to be executed by an HPX thread during the execution of
    /// \a hpx::finalize() after any of the registered pre-shutdown functions
    /// are executed (see: \a hpx::register_pre_shutdown_function()).
    ///
    /// \param f  [in] The function to be registered to run by an HPX thread as
    ///           a shutdown function.
    ///
    /// \note If this function is called while the shutdown functions are
    ///       being executed, or after that point, it will raise a invalid_status
    ///       exception.
    ///
    /// \see    \a hpx::register_pre_shutdown_function()
    HPX_API_EXPORT void register_shutdown_function(shutdown_function_type const& f);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Test whether the runtime system is currently being started.
    ///
    /// This function returns whether the runtime system is currently being
    /// started or not, e.g. whether the current state of the runtime system is
    /// \a hpx::state_startup
    ///
    /// \note   This function needs to be executed on a HPX-thread. It will
    ///         return false otherwise.
    HPX_API_EXPORT bool is_starting();

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Test whether the runtime system is currently running.
    ///
    /// This function returns whether the runtime system is currently running
    /// or not, e.g.  whether the current state of the runtime system is
    /// \a hpx::state_running
    ///
    /// \note   This function needs to be executed on a HPX-thread. It will
    ///         return false otherwise.
    HPX_API_EXPORT bool is_running();

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Test whether the runtime system is currently stopped.
    ///
    /// This function returns whether the runtime system is currently stopped
    /// or not, e.g.  whether the current state of the runtime system is
    /// \a hpx::state_stopped
    ///
    /// \note   This function needs to be executed on a HPX-thread. It will
    ///         return false otherwise.
    HPX_API_EXPORT bool is_stopped();

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Test whether the runtime system is currently being shut down.
    ///
    /// This function returns whether the runtime system is currently being
    /// shut down or not, e.g.  whether the current state of the runtime system
    /// is \a hpx::state_stopped or \a hpx::state_shutdown
    ///
    /// \note   This function needs to be executed on a HPX-thread. It will
    ///         return false otherwise.
    HPX_API_EXPORT bool is_stopped_or_shutting_down();

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the name of the calling thread.
    ///
    /// This function returns the name of the calling thread. This name uniquely
    /// identifies the thread in the context of HPX. If the function is called
    /// while no HPX runtime system is active, the result will be "<unknown>".
    HPX_API_EXPORT std::string get_thread_name();

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the number of worker OS- threads used to execute HPX
    ///        threads
    ///
    /// This function returns the number of OS-threads used to execute HPX
    /// threads. If the function is called while no HPX runtime system is active,
    /// it will return zero.
    HPX_API_EXPORT std::size_t get_num_worker_threads();

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the system uptime measure on the thread executing this call.
    ///
    /// This function returns the system uptime measured in nanoseconds for the
    /// thread executing this call. If the function is called while no HPX
    /// runtime system is active, it will return zero.
    HPX_API_EXPORT boost::uint64_t get_system_uptime();

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the id of the locality where the object referenced by the
    ///        given id is currently located on
    ///
    /// The function hpx::get_colocation_id() returns the id of the locality
    /// where the given object is currently located.
    ///
    /// \param id [in] The id of the object to locate.
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// \see    \a hpx::get_colocation_id()
    HPX_API_EXPORT naming::id_type get_colocation_id_sync(
        naming::id_type const& id, error_code& ec = throws);

    /// \brief Asynchronously return the id of the locality where the object
    ///        referenced by the given id is currently located on
    ///
    /// \see    \a hpx::get_colocation_id_sync()
    HPX_API_EXPORT lcos::future<naming::id_type> get_colocation_id(
        naming::id_type const& id);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Start all active performance counters, optionally naming the
    ///        section of code
    ///
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// \note     The active counters are those which have been specified on
    ///           the command line while executing the application (see command
    ///           line option \--hpx:print-counter)
    HPX_API_EXPORT void start_active_counters(error_code& ec = throws);

    /// \brief Resets all active performance counters.
    ///
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// \note     The active counters are those which have been specified on
    ///           the command line while executing the application (see command
    ///           line option \--hpx:print-counter)
    HPX_API_EXPORT void reset_active_counters(error_code& ec = throws);

    /// \brief Stop all active performance counters.
    ///
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// \note     The active counters are those which have been specified on
    ///           the command line while executing the application (see command
    ///           line option \--hpx:print-counter)
    HPX_API_EXPORT void stop_active_counters(error_code& ec = throws);

    /// \brief Evaluate and output all active performance counters, optionally
    ///        naming the point in code marked by this function.
    ///
    /// \param reset       [in] this is an optional flag allowing to reset
    ///                    the counter value after it has been evaluated.
    /// \param description [in] this is an optional value naming the point in
    ///                    the code marked by the call to this function.
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// \note     The output generated by this function is redirected to the
    ///           destination specified by the corresponding command line
    ///           options (see \--hpx:print-counter-destination).
    ///
    /// \note     The active counters are those which have been specified on
    ///           the command line while executing the application (see command
    ///           line option \--hpx:print-counter)
    HPX_API_EXPORT void evaluate_active_counters(bool reset = false,
        char const* description = 0, error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Create an instance of a binary filter plugin
    ///
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    HPX_API_EXPORT serialization::binary_filter* create_binary_filter(
        char const* binary_filter_type, bool compress,
        serialization::binary_filter* next_filter = 0, error_code& ec = throws);

#if defined(HPX_HAVE_SODIUM)
    namespace components { namespace security
    {
        class certificate;
        class certificate_signing_request;
        class parcel_suffix;
        class hash;

        template <typename T> class signed_type;
        typedef signed_type<certificate> signed_certificate;
        typedef signed_type<certificate_signing_request>
            signed_certificate_signing_request;
        typedef signed_type<parcel_suffix> signed_parcel_suffix;
    }}

#if defined(HPX_HAVE_SECURITY)
    /// \brief Return the certificate for this locality
    ///
    /// \returns This function returns the signed certificate for this locality.
    HPX_API_EXPORT components::security::signed_certificate const&
        get_locality_certificate(error_code& ec = throws);

    /// \brief Return the certificate for the given locality
    ///
    /// \param id The id representing the locality for which to retrieve
    ///           the signed certificate.
    ///
    /// \returns This function returns the signed certificate for the locality
    ///          identified by the parameter \a id.
    HPX_API_EXPORT components::security::signed_certificate const&
        get_locality_certificate(boost::uint32_t locality_id, error_code& ec = throws);

    /// \brief Add the given certificate to the certificate store of this locality.
    ///
    /// \param cert The certificate to add to the certificate store of this
    ///             locality
    HPX_API_EXPORT void add_locality_certificate(
        components::security::signed_certificate const& cert,
        error_code& ec = throws);
#endif
#endif
}

// Including declarations of various API function declarations
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/basename_registration.hpp>
#include <hpx/runtime/trigger_lco.hpp>
#include <hpx/runtime/get_locality_name.hpp>
#include <hpx/runtime/get_locality_id.hpp>
#include <hpx/runtime/get_config_entry.hpp>
#include <hpx/runtime/set_parcel_write_handler.hpp>
#include <hpx/runtime/get_os_thread_count.hpp>
#include <hpx/runtime/get_worker_thread_num.hpp>
#include <hpx/runtime/message_handler_fwd.hpp>

#include <hpx/lcos/async_fwd.hpp>
#include <hpx/lcos/async_callback_fwd.hpp>

#endif

