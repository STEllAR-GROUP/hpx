//  Copyright (c) 2016-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/futures/future_fwd.hpp>
#include <hpx/runtime/naming_fwd.hpp>

#include <vector>

namespace hpx { namespace compute { namespace host {
    struct HPX_EXPORT target;

    HPX_EXPORT std::vector<target> get_local_targets();
    HPX_EXPORT hpx::future<std::vector<target>> get_targets(
        hpx::id_type const& locality);
}}}    // namespace hpx::compute::host
