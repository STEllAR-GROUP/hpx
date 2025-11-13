//  Copyright (c) 2019 Thomas Heller
//  Copyright (c) 2022-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <hpx/assertion/source_location.hpp>

#include <string>

namespace hpx::assertion::detail {

    /// \cond NOINTERNAL
    HPX_CXX_EXPORT HPX_CORE_EXPORT void handle_assert(
        hpx::source_location const& loc, char const* expr,
        std::string const& msg) noexcept;
    /// \endcond
}    // namespace hpx::assertion::detail
