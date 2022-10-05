//  Copyright (c) 2022 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/likwid/likwid_init.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace likwid {
    struct likwid_init
    {
        likwid_init()
        {
            // Initalise likwid marker API
            likwid_markerInit();

            // thread init helper
            likwid_thread_init();
        }
        ~likwid_init()
        {
            likwid_markerClose();
        }
    };
    likwid_init likwid_init_helper;
}}}