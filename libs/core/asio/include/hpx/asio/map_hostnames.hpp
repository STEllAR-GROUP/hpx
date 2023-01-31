//  Copyright (c) 2007-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/functional/function.hpp>

#include <cstdint>
#include <string>

#if defined(HPX_MSVC_WARNING_PRAGMA)
#pragma warning(push)
#pragma warning(disable : 4251)
#endif

namespace hpx::util {

    ///////////////////////////////////////////////////////////////////////////
    // Try to map a given host name based on the list of mappings read from a
    // file
    struct HPX_CORE_EXPORT map_hostnames
    {
        using transform_function_type =
            hpx::function<std::string(std::string const&)>;

        explicit map_hostnames(bool debug = false) noexcept
          : ipv4_(false)
          , debug_(debug)
        {
        }

        void use_suffix(std::string const& suffix)
        {
            suffix_ = suffix;
        }

        void use_prefix(std::string const& prefix)
        {
            prefix_ = prefix;
        }

        void use_transform(transform_function_type const& f)
        {
            transform_ = f;
        }

        void force_ipv4(bool f) noexcept
        {
            ipv4_ = f;
        }

        [[nodiscard]] std::string map(
            std::string host_name, std::uint16_t port) const;

    private:
        transform_function_type transform_;
        std::string suffix_;
        std::string prefix_;
        bool ipv4_;
        bool debug_;
    };
}    // namespace hpx::util

#if defined(HPX_MSVC_WARNING_PRAGMA)
#pragma warning(pop)
#endif
