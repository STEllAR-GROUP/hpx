//  Copyright (c) 2019-2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <filesystem>
#include <string>
#include <system_error>

namespace hpx { namespace filesystem { namespace detail {
    HPX_CORE_EXPORT std::filesystem::path initial_path();
    HPX_CORE_EXPORT std::string basename(std::filesystem::path const& p);
    HPX_CORE_EXPORT std::filesystem::path canonical(
        std::filesystem::path const& p, std::filesystem::path const& base);
    HPX_CORE_EXPORT std::filesystem::path canonical(
        std::filesystem::path const& p, std::filesystem::path const& base,
        std::error_code& ec);
}}}    // namespace hpx::filesystem::detail
