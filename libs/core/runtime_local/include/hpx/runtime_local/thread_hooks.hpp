//  Copyright (c) 2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/threading_base/callback_notifier.hpp>

namespace hpx {
    /// Retrieve the currently installed start handler function. This is a
    /// function that will be called by HPX for each newly created thread that
    /// is made known to the runtime. HPX stores exactly one such function
    /// reference, thus the caller needs to make sure any newly registered
    /// start function chains into the previous one (see
    /// \a register_thread_on_start_func).
    ///
    /// \returns The currently installed error handler function.
    ///
    /// \note This function can be called before the HPX runtime is initialized.
    ///
    HPX_CORE_EXPORT threads::policies::callback_notifier::on_startstop_type
    get_thread_on_start_func();

    /// Retrieve the currently installed stop handler function. This is a
    /// function that will be called by HPX for each newly created thread that
    /// is made known to the runtime. HPX stores exactly one such function
    /// reference, thus the caller needs to make sure any newly registered
    /// stop function chains into the previous one (see
    /// \a register_thread_on_stop_func).
    ///
    /// \returns The currently installed error handler function.
    ///
    /// \note This function can be called before the HPX runtime is initialized.
    ///
    HPX_CORE_EXPORT threads::policies::callback_notifier::on_startstop_type
    get_thread_on_stop_func();

    /// Retrieve the currently installed error handler function. This is a
    /// function that will be called by HPX for each newly created thread that
    /// is made known to the runtime. HPX stores exactly one such function
    /// reference, thus the caller needs to make sure any newly registered
    /// error function chains into the previous one (see
    /// \a register_thread_on_error_func).
    ///
    /// \returns The currently installed error handler function.
    ///
    /// \note This function can be called before the HPX runtime is initialized.
    ///
    HPX_CORE_EXPORT threads::policies::callback_notifier::on_error_type
    get_thread_on_error_func();

    /// Set the currently installed start handler function. This is a
    /// function that will be called by HPX for each newly created thread that
    /// is made known to the runtime. HPX stores exactly one such function
    /// reference, thus the caller needs to make sure any newly registered
    /// start function chains into the previous one (see
    /// \a get_thread_on_start_func).
    ///
    /// \param f The function to install as the new start handler.
    ///
    /// \returns The previously registered function of this category. It is
    ///          the user's responsibility to call that function if the
    ///          callback is invoked by HPX.
    ///
    /// \note This function can be called before the HPX runtime is initialized.
    ///
    HPX_CORE_EXPORT threads::policies::callback_notifier::on_startstop_type
    register_thread_on_start_func(
        threads::policies::callback_notifier::on_startstop_type&& f);

    /// Set the currently installed stop handler function. This is a
    /// function that will be called by HPX for each newly created thread that
    /// is made known to the runtime. HPX stores exactly one such function
    /// reference, thus the caller needs to make sure any newly registered
    /// stop function chains into the previous one (see
    /// \a get_thread_on_stop_func).
    ///
    /// \param f The function to install as the new stop handler.
    ///
    /// \returns The previously registered function of this category. It is
    ///          the user's responsibility to call that function if the
    ///          callback is invoked by HPX.
    ///
    /// \note This function can be called before the HPX runtime is initialized.
    ///
    HPX_CORE_EXPORT threads::policies::callback_notifier::on_startstop_type
    register_thread_on_stop_func(
        threads::policies::callback_notifier::on_startstop_type&& f);

    /// Set the currently installed error handler function. This is a
    /// function that will be called by HPX for each newly created thread that
    /// is made known to the runtime. HPX stores exactly one such function
    /// reference, thus the caller needs to make sure any newly registered
    /// error function chains into the previous one (see
    /// \a get_thread_on_error_func).
    ///
    /// \param f The function to install as the new error handler.
    ///
    /// \returns The previously registered function of this category. It is
    ///          the user's responsibility to call that function if the
    ///          callback is invoked by HPX.
    ///
    /// \note This function can be called before the HPX runtime is initialized.
    ///
    HPX_CORE_EXPORT threads::policies::callback_notifier::on_error_type
    register_thread_on_error_func(
        threads::policies::callback_notifier::on_error_type&& f);
}    // namespace hpx
