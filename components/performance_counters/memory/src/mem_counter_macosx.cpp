// Copyright (c) 2013 Jeroen Habraken
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(__APPLE__)

#include <hpx/modules/errors.hpp>

#include <cstdint>

#include <mach/mach.h>
#include <mach/task.h>

namespace hpx { namespace performance_counters { namespace memory
{
    ///////////////////////////////////////////////////////////////////////////
    // returns virtual memory value
    std::uint64_t read_psm_virtual(bool)
    {
        struct task_basic_info t_info;
        mach_msg_type_number_t t_info_count = TASK_BASIC_INFO_COUNT;

        if (task_info(mach_task_self(),
                TASK_BASIC_INFO,
                reinterpret_cast<task_info_t>(&t_info),
                &t_info_count) != KERN_SUCCESS)
        {
            HPX_THROW_EXCEPTION(kernel_error,
                "hpx::performance_counters::memory::read_psm_virtual",
                "task_info failed");

            return std::uint64_t(-1);
        }

        return t_info.virtual_size;
    }

    ///////////////////////////////////////////////////////////////////////////
    // returns resident memory value
    std::uint64_t read_psm_resident(bool)
    {
        struct task_basic_info t_info;
        mach_msg_type_number_t t_info_count = TASK_BASIC_INFO_COUNT;

        if (task_info(mach_task_self(),
                TASK_BASIC_INFO,
                reinterpret_cast<task_info_t>(&t_info),
                &t_info_count) != KERN_SUCCESS)
        {
            HPX_THROW_EXCEPTION(kernel_error,
                "hpx::performance_counters::memory::read_psm_virtual",
                "task_info failed");

            return std::uint64_t(-1);
        }

        return t_info.resident_size;
    }
}}}

#endif
