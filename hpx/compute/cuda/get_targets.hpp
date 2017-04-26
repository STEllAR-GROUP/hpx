///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#ifndef HPX_COMPUTE_CUDA_GET_TARGETS_HPP
#define HPX_COMPUTE_CUDA_GET_TARGETS_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CUDA)
#include <hpx/lcos_fwd.hpp>
#include <hpx/runtime/naming_fwd.hpp>

#include <vector>

namespace hpx { namespace compute { namespace cuda
{
    struct HPX_EXPORT target;

    HPX_EXPORT std::vector<target> get_local_targets();
    HPX_EXPORT hpx::future<std::vector<target> >
        get_targets(hpx::id_type const& locality);
}}}

#endif
#endif
