//  Copyright (c) 2019-2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/filesystem.hpp>

#include <filesystem>

namespace hpx { namespace filesystem { namespace detail {
    std::filesystem::path initial_path()
    {
        static std::filesystem::path ip = std::filesystem::current_path();
        return ip;
    }

    std::string basename(std::filesystem::path const& p)
    {
        return p.stem().string();
    }

    std::filesystem::path canonical(
        std::filesystem::path const& p, std::filesystem::path const& base)
    {
        if (p.is_relative())
        {
            return std::filesystem::canonical(base / p);
        }
        else
        {
            return std::filesystem::canonical(p);
        }
    }

    std::filesystem::path canonical(std::filesystem::path const& p,
        std::filesystem::path const& base, std::error_code& ec)
    {
        if (p.is_relative())
        {
            return std::filesystem::canonical(base / p, ec);
        }
        else
        {
            return std::filesystem::canonical(p, ec);
        }
    }
}}}    // namespace hpx::filesystem::detail
