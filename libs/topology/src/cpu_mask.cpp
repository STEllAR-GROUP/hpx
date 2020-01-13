//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/topology/cpu_mask.hpp>

#include <iomanip>
#include <sstream>
#include <string>

namespace hpx { namespace threads {
    std::string to_string(mask_cref_type val)
    {
        std::ostringstream ostr;
        ostr << std::hex << HPX_CPU_MASK_PREFIX << val;
        return ostr.str();
    }
}}    // namespace hpx::threads
