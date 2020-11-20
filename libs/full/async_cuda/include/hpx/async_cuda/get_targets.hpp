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
#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
#include <hpx/naming_base/id_type.hpp>
#endif

#include <vector>

namespace hpx { namespace cuda { namespace experimental {
    struct HPX_EXPORT target;

    HPX_EXPORT std::vector<target> get_local_targets();
#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
    HPX_EXPORT hpx::future<std::vector<target>> get_targets(
        hpx::id_type const& locality);
#endif
    HPX_EXPORT void print_local_targets();

}}}    // namespace hpx::cuda::experimental
