///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#include <hpx/compute/host/target.hpp>
#include <hpx/resource_partitioner/detail/partitioner.hpp>
#include <hpx/runtime_local/get_os_thread_count.hpp>
#include <hpx/runtime_local/runtime_local.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/topology/topology.hpp>

#if defined(HPX_HAVE_MORE_THAN_64_THREADS)
#if defined(HPX_HAVE_MAX_CPU_COUNT)
#include <hpx/serialization/bitset.hpp>
#else
#include <hpx/serialization/dynamic_bitset.hpp>
#endif
#endif

#include <cstddef>
#include <utility>

namespace hpx { namespace compute { namespace host {
    std::pair<std::size_t, std::size_t> target::num_pus() const
    {
        auto& rp = hpx::resource::get_partitioner();
        std::size_t num_os_threads = hpx::get_os_thread_count();

        hpx::threads::mask_type mask = native_handle().get_device();
        std::size_t mask_size = hpx::threads::mask_size(mask);

        std::size_t num_thread = 0;
        for (/**/; num_thread != num_os_threads; ++num_thread)
        {
            if (hpx::threads::bit_and(
                    mask, rp.get_pu_mask(num_thread), mask_size))
            {
                break;
            }
        }
        return std::make_pair(num_thread, hpx::threads::count(mask));
    }

    void target::serialize(serialization::input_archive& ar, const unsigned int)
    {
#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
        ar >> handle_.mask_ >> locality_;
#else
        ar >> handle_.mask_;
#endif
    }

    void target::serialize(
        serialization::output_archive& ar, const unsigned int)
    {
#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
        ar << handle_.mask_ << locality_;
#else
        ar << handle_.mask_;
#endif
    }
}}}    // namespace hpx::compute::host
