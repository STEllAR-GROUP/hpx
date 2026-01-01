//  Copyright (c) 2022 Srinivas Yadav
//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_MODULE_LIKWID)
#include <hpx/modules/runtime_local.hpp>

#include <cstddef>

#include <likwid.h>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::likwid {

    void likwid_thread_init()
    {
        auto prev = hpx::get_thread_on_start_func();
        hpx::register_thread_on_start_func(
            [prev = HPX_MOVE(prev)](std::size_t local_thread_num,
                std::size_t global_thread_num, char const* pool_name,
                char const* name_postfix) {
                likwid_markerThreadInit();
                if (!prev.empty())
                {
                    prev(local_thread_num, global_thread_num, pool_name,
                        name_postfix);
                }
            });
    }

    struct likwid_init
    {
        likwid_init()
        {
            // Initialize likwid marker API
            likwid_markerInit();

            // thread init helper
            likwid_thread_init();
        }

        ~likwid_init() noexcept
        {
            likwid_markerClose();
        }
    };

    likwid_init likwid_init_helper;
}    // namespace hpx::likwid

#endif
