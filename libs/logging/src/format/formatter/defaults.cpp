// defaults.cpp

// Boost Logging library
//
// Author: John Torjo, www.torjo.com
//
// Copyright (C) 2007 John Torjo (see www.torjo.com for email)
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
//
// See http://www.boost.org for updates, documentation, and revision history.
// See http://www.torjo.com/log2/ for more details

#include <hpx/logging/format/formatters.hpp>

#include <hpx/config.hpp>
#include <hpx/format.hpp>

#include <cstdint>
#include <memory>
#include <ostream>

namespace hpx { namespace util { namespace logging { namespace formatter {

    idx::~idx() = default;

    struct idx_impl : idx
    {
        idx_impl()
          : value(0ull)
        {
        }

        void operator()(std::ostream& to) const override
        {
            util::format_to(to, "{:016x}", ++value);
        }

    private:
        mutable std::uint64_t value;
    };

    std::shared_ptr<idx> idx::make()
    {
        return std::make_shared<idx_impl>();
    }

}}}}    // namespace hpx::util::logging::formatter
