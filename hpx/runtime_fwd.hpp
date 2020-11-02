//  Copyright (c) 2007-2018 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file runtime_fwd.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/runtime_local/runtime_local_fwd.hpp>

#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
#include <hpx/modules/naming_base.hpp>
#include <hpx/runtime/basename_registration_fwd.hpp>
#include <hpx/runtime/find_all_localities.hpp>
#include <hpx/runtime/get_locality_name.hpp>
#include <hpx/runtime/set_parcel_write_handler.hpp>
#include <hpx/runtime_local/get_locality_id.hpp>
#include <hpx/runtime_local/get_num_all_localities.hpp>

#include <cstddef>
#include <cstdint>
#include <string>

namespace hpx {
    /// The function \a get_locality returns a reference to the locality prefix
    HPX_EXPORT naming::gid_type const& get_locality();

    /// \cond NOINTERNAL
    namespace util {
        struct binary_filter;
    }    // namespace util

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
    HPX_EXPORT void start_active_counters(error_code& ec = throws);

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
    HPX_EXPORT void reset_active_counters(error_code& ec = throws);

    /// \brief Re-initialize all active performance counters.
    ///
    /// \param reset [in] Reset the current values before re-initializing
    ///           counters (default: true)
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
    HPX_EXPORT void reinit_active_counters(
        bool reset = true, error_code& ec = throws);

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
    HPX_EXPORT void stop_active_counters(error_code& ec = throws);

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
    HPX_EXPORT void evaluate_active_counters(bool reset = false,
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
    HPX_EXPORT serialization::binary_filter* create_binary_filter(
        char const* binary_filter_type, bool compress,
        serialization::binary_filter* next_filter = nullptr,
        error_code& ec = throws);
}    // namespace hpx
#endif
