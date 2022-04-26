//  Copyright (c) 2016-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/compute_local/host/get_targets.hpp>

#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
#include <hpx/futures/future_fwd.hpp>
#include <hpx/modules/compute_local.hpp>
#include <hpx/modules/naming.hpp>

#include <vector>

namespace hpx::compute::host::distributed {

    struct HPX_EXPORT target;

    HPX_EXPORT hpx::future<std::vector<target>> get_targets(
        hpx::id_type const& locality);

    namespace detail {

        HPX_EXPORT std::vector<host::distributed::target> get_remote_targets(
            std::vector<host::target> const& targets);
    }
}    // namespace hpx::compute::host::distributed

#endif
