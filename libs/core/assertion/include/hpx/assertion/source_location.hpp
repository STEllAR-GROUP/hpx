//  Copyright (c) 2019 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config/export_definitions.hpp>

#include <iosfwd>

namespace hpx { namespace assertion {
    /// This contains the location information where \a HPX_ASSERT has been
    /// called
    struct source_location
    {
        const char* file_name;
        unsigned line_number;
        const char* function_name;
    };
    HPX_CORE_EXPORT std::ostream& operator<<(
        std::ostream& os, source_location const& loc);
}}    // namespace hpx::assertion
