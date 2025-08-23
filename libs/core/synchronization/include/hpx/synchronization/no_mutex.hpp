//  Copyright (c) 2007-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file no_mutex.hpp
/// \page hpx::no_mutex
/// \headerfile hpx/mutex.hpp

#pragma once

#include <hpx/config.hpp>

namespace hpx {

    /// \c no_mutex class can be used in cases where the shared data between
    /// multiple threads can be accessed simultaneously without causing
    /// inconsistencies.
    struct no_mutex
    {
        static constexpr void lock() noexcept {}

        static constexpr bool try_lock() noexcept
        {
            return true;
        }

        static constexpr void unlock() noexcept {}
    };
}    // namespace hpx
