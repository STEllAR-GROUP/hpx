//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file cuda/target_distribution_policy.hpp

#if !defined(HPX_COMPUTE_CUDA_TARGET_DISTRIBUTION_POLICY)
#define HPX_COMPUTE_CUDA_TARGET_DISTRIBUTION_POLICY

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CUDA) && defined(__CUDACC__)

#include <hpx/compute/cuda/target.hpp>
#include <hpx/compute/target_distribution_policy.hpp>

namespace hpx { namespace compute { namespace cuda
{
    /// A predefined instance of the \a target_distribution_policy for CUDA.
    /// It will represent all local CUDA devices and will place all items to
    /// create here.
    static compute::target_distribution_policy<cuda::target> const target_layout;
}}}

#endif
#endif
