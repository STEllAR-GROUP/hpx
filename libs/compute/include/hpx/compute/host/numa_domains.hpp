///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#ifndef HPX_COMPUTE_HOST_NUMA_DOMAINS_HPP
#define HPX_COMPUTE_HOST_NUMA_DOMAINS_HPP

#include <hpx/compute/host/target.hpp>

#include <vector>

namespace hpx { namespace compute { namespace host {
    HPX_EXPORT std::vector<target> numa_domains();
}}}    // namespace hpx::compute::host

#endif
