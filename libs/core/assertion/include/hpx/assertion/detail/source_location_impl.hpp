//  Copyright (c) 2019 Thomas Heller
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assertion/source_location.hpp>

#include <ostream>

#if !defined(HPX_ASSERTION_INLINE)
#define HPX_ASSERTION_INLINE inline
#endif

namespace hpx {

    HPX_ASSERTION_INLINE std::ostream& operator<<(
        std::ostream& os, hpx::source_location const& loc)
    {
        os << loc.file_name() << ":" << loc.line() << ": "
           << loc.function_name();
        return os;
    }
}    // namespace hpx

#undef HPX_ASSERTION_INLINE
