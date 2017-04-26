//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file runtime_fwd.hpp

#ifndef HPX_RUNTIME_FWD_HPP
#define HPX_RUNTIME_FWD_HPP

#include <hpx/config.hpp>
#include <hpx/exception_fwd.hpp>
#include <hpx/runtime/basename_registration.hpp>
#include <hpx/runtime/config_entry.hpp>
#include <hpx/runtime/find_localities.hpp>
#include <hpx/runtime/get_colocation_id.hpp>
#include <hpx/runtime/get_locality_id.hpp>
#include <hpx/runtime/get_locality_name.hpp>
#include <hpx/runtime/get_num_localities.hpp>
#include <hpx/runtime/get_os_thread_count.hpp>
#include <hpx/runtime/get_thread_name.hpp>
#include <hpx/runtime/get_worker_thread_num.hpp>
#include <hpx/runtime/naming_fwd.hpp>
#include <hpx/runtime/report_error.hpp>
#include <hpx/runtime/runtime_fwd.hpp>
#include <hpx/runtime/runtime_mode.hpp>
#include <hpx/runtime/set_parcel_write_handler.hpp>
#include <hpx/runtime/shutdown_function.hpp>
#include <hpx/runtime/startup_function.hpp>
#include <hpx/util/function.hpp>
#include <hpx/util_fwd.hpp>


#include <cstddef>
#include <cstdint>
#include <string>

namespace hpx
{
    /// Register the current kernel thread with HPX, this should be done once
    /// for each external OS-thread intended to invoke HPX functionality.
    /// Calling this function more than once will silently fail.
    HPX_API_EXPORT bool register_thread(runtime* rt, char const* name,
        error_code& ec = throws);

    /// Unregister the thread from HPX, this should be done once in
    /// the end before the external thread exists.
    HPX_API_EXPORT void unregister_thread(runtime* rt);

    /// The function \a get_locality returns a reference to the locality prefix
    HPX_API_EXPORT naming::gid_type const& get_locality();

    /// The function \a get_runtime_instance_number returns a unique number
    /// associated with the runtime instance the current thread is running in.
    HPX_API_EXPORT std::size_t get_runtime_instance_number();

    /// Register a function to be called during system shutdown
    HPX_API_EXPORT bool register_on_exit(util::function_nonser<void()> const&);

    /// \cond NOINTERNAL
    namespace util
    {
        struct binary_filter;

        /// \brief Expand INI variables in a string
        HPX_API_EXPORT std::string expand(std::string const& expand);

        /// \brief Expand INI variables in a string
        HPX_API_EXPORT void expand(std::string& expand);
    }

    ///////////////////////////////////////////////////////////////////////////
    HPX_API_EXPORT bool is_scheduler_numa_sensitive();

    ///////////////////////////////////////////////////////////////////////////
    HPX_API_EXPORT util::runtime_configuration const& get_config();

    ///////////////////////////////////////////////////////////////////////////
    HPX_API_EXPORT hpx::util::io_service_pool* get_thread_pool(
        char const* name, char const* pool_name_suffix = "");

    /// \endcond

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
    HPX_API_EXPORT std::uint64_t get_system_uptime();

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
        char const* description = nullptr, error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Create an instance of a binary filter plugin
    ///
    /// \param binary_filter_type [in] The type of the binary filter to create
    /// \param compress     [in] The created filter should support compression
    /// \param next_filter  [in] Use this as the filter to dispatch the
    ///                     invocation into.
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
        serialization::binary_filter* next_filter = nullptr,
        error_code& ec = throws);
}

#endif
