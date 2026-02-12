//  Copyright (c) 2016-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <vector>

namespace hpx::compute::host {

    HPX_CXX_CORE_EXPORT struct HPX_CORE_EXPORT target;

    HPX_CXX_CORE_EXPORT HPX_CORE_EXPORT std::vector<target> get_local_targets();
}    // namespace hpx::compute::host
