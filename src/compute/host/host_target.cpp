///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#include <hpx/compute/host/target.hpp>
#include <hpx/runtime.hpp>
#include <hpx/runtime/get_os_thread_count.hpp>
#include <hpx/runtime/resource/detail/partitioner.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/runtime/threads/topology.hpp>

#if defined(HPX_HAVE_MORE_THAN_64_THREADS)
# if defined(HPX_HAVE_MAX_CPU_COUNT)
#  include <hpx/runtime/serialization/bitset.hpp>
# else
#  include <hpx/runtime/serialization/dynamic_bitset.hpp>
# endif
#endif

#include <cstddef>
#include <utility>

namespace hpx { namespace compute { namespace host
{
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
        ar >> handle_.mask_ >> locality_;
    }

    void target::serialize(serialization::output_archive& ar, const unsigned int)
    {
        ar << handle_.mask_ << locality_;
    }
}}}
