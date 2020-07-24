//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/datastructures/any.hpp>
#include <hpx/serialization/serializable_any.hpp>

namespace hpx {
    using hpx::util::any;
    using hpx::util::any_nonser;
    using hpx::util::bad_any_cast;
    using hpx::util::make_any;
    using hpx::util::make_any_nonser;
    using hpx::util::make_unique_any_nonser;
    using hpx::util::unique_any_nonser;
}    // namespace hpx
