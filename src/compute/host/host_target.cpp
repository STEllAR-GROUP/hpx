///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#include <hpx/compute/host/target.hpp>
#include <hpx/runtime.hpp>
#include <hpx/runtime/get_os_thread_count.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/runtime/threads/topology.hpp>

#include <utility>

namespace hpx { namespace compute { namespace host
{
    std::pair<std::size_t, std::size_t> target::num_pus() const
    {
        auto const& topo = hpx::threads::get_topology();
        auto & tm = hpx::get_runtime().get_thread_manager();
        std::size_t num_os_threads = hpx::get_os_thread_count();
        hpx::threads::mask_type mask = native_handle().get_device();
        std::size_t mask_size = hpx::threads::mask_size(mask);

        std::size_t num_thread = 0;
        for(/**/; num_thread != num_os_threads; ++num_thread)
        {
            if(hpx::threads::bit_and(
                    mask, tm.get_pu_mask(topo, num_thread), mask_size))
            {
                break;
            }
        }
        return std::make_pair(num_thread, hpx::threads::count(mask));
    }
}}}
