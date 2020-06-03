//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING)
#include <hpx/modules/errors.hpp>
#include <hpx/runtime/parcelset_fwd.hpp>

#include <cstddef>

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    /// Register an instance of a message handler plugin
    ///
    /// The function hpx::register_message_handler() registers an instance of a
    /// message handler plugin based on the parameters specified.
    ///
    /// \param message_handler_type
    /// \param action   [in] The name of the action for which a plugin should
    ///                 be registered
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    HPX_EXPORT void register_message_handler(
        char const* message_handler_type, char const* action,
        error_code& ec = throws);

    /// \brief Create an instance of a message handler plugin
    ///
    /// The function hpx::create_message_handler() creates an instance of a
    /// message handler plugin based on the parameters specified.
    ///
    /// \param message_handler_type
    /// \param action
    /// \param pp
    /// \param num_messages
    /// \param interval
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    HPX_EXPORT parcelset::policies::message_handler* create_message_handler(
        char const* message_handler_type, char const* action,
        parcelset::parcelport* pp, std::size_t num_messages,
        std::size_t interval, error_code& ec = throws);

}

#endif
