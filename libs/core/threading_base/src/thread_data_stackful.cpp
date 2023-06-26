//  Copyright (c) 2019-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/allocator_support/internal_allocator.hpp>
#include <hpx/modules/logging.hpp>
#include <hpx/threading_base/thread_data.hpp>

////////////////////////////////////////////////////////////////////////////////
namespace hpx::threads {

    util::internal_allocator<thread_data_stackful>
        thread_data_stackful::thread_alloc_;

#if !defined(HPX_HAVE_LOGGING)
    thread_data_stackful::~thread_data_stackful() = default;
#else
    thread_data_stackful::~thread_data_stackful()
    {
        LTM_(debug).format(
            "~thread_data_stackful({}), description({}), phase({})", this,
            this->get_description(),
            this->thread_data_stackful::get_thread_phase());
    }
#endif
}    // namespace hpx::threads
