//  Copyright (c) 2019 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/assertion/source_location.hpp>

#include <ostream>

namespace hpx {

    std::ostream& operator<<(std::ostream& os, hpx::source_location const& loc)
    {
        os << loc.file_name() << ":" << loc.line() << ": "
           << loc.function_name();
        return os;
    }
}    // namespace hpx
