//  Copyright (c) 2022 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <likwid.h>
#include <hpx/modules/runtime_local.hpp>

#include <iostream>
///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace likwid {
    struct likwid_init;

    extern likwid_init likwid_init_helper;

    void likwid_thread_init()
    {
        auto prev = hpx::get_thread_on_start_func();
        hpx::register_thread_on_start_func
        (
            [prev](std::size_t local_thread_num,std::size_t global_thread_num,
                    char const* pool_name, char const* name_postfix)
            {
                likwid_markerThreadInit();
                if (!prev.empty())
                    prev(local_thread_num, global_thread_num, pool_name, name_postfix);
            }
        );
    }
}}}