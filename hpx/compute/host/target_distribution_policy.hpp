//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file host/target_distribution_policy.hpp

#if !defined(HPX_COMPUTE_HOST_TARGET_DISTRIBUTION_POLICY)
#define HPX_COMPUTE_HOST_TARGET_DISTRIBUTION_POLICY

#include <hpx/config.hpp>

#include <hpx/compute/target_distribution_policy.hpp>
#include <hpx/compute/host/target.hpp>

namespace hpx { namespace compute { namespace host
{
    /// A predefined instance of the \a target_distribution_policy for
    /// localities. It will represent all NUMA domains of the given locality
    /// and will place all items to create here.
    static compute::target_distribution_policy<host::target> const target_layout;
}}}

#endif
