//  Copyright (c) 2019 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/local/config.hpp>

#include <hpx/assertion/source_location.hpp>

#include <string>
#include <utility>

namespace hpx { namespace assertion { namespace detail {
    /// \cond NOINTERNAL
    HPX_LOCAL_EXPORT void handle_assert(source_location const& loc,
        const char* expr, std::string const& msg) noexcept;
    /// \endcond
}}}    // namespace hpx::assertion::detail
