//  Copyright (c) 2019 Thomas Heller
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assertion/export_definitions.hpp>
#include <hpx/assertion/source_location.hpp>

#include <string>
#include <utility>

namespace hpx::assertion {

    /// \cond NOINTERNAL
    namespace detail {
        // The signature for an assertion handler
        using assertion_handler = void (*)(hpx::source_location const& loc,
            const char* expr, std::string const& msg);

        HPX_CORE_ASSERTION_EXPORT void handle_assert(
            hpx::source_location const& loc, const char* expr,
            std::string const& msg) noexcept;
    }    // namespace detail
    /// \endcond

    /// Set the assertion handler to be used within a program. If the handler
    /// has been set already once, the call to this function will be ignored.
    /// \note This function is not thread safe
    HPX_CORE_ASSERTION_EXPORT void set_assertion_handler(
        detail::assertion_handler handler);
}    // namespace hpx::assertion

//#include <hpx/assertion/detail/evaluate_assert_impl.hpp>
