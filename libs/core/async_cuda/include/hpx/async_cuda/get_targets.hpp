///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/futures.hpp>

#include <vector>

namespace hpx { namespace cuda { namespace experimental {
    struct HPX_CORE_EXPORT target;

    HPX_CORE_EXPORT std::vector<target> get_local_targets();
    HPX_CORE_EXPORT void print_local_targets();

}}}    // namespace hpx::cuda::experimental
