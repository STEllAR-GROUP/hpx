//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/thread_executors/thread_executor.hpp>

#if defined(HPX_HAVE_THREAD_EXECUTORS_COMPATIBILITY)

#include <hpx/modules/threadmanager.hpp>
#include <hpx/resource_partitioner/detail/partitioner.hpp>
#include <hpx/static_reinit/reinitializable_static.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/topology/topology.hpp>

#include <cstddef>
#include <mutex>

namespace hpx { namespace threads { namespace detail {
    mask_cref_type executor_base::get_pu_mask(
        topology const& /* topology */, std::size_t num_thread) const
    {
        auto& rp = hpx::resource::get_partitioner();
        return rp.get_pu_mask(num_thread);
    }
}}}    // namespace hpx::threads::detail
#endif
