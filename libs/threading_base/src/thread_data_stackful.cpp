//  Copyright (c) 2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/allocator_support/internal_allocator.hpp>
#include <hpx/modules/logging.hpp>
#include <hpx/threading_base/thread_data.hpp>

////////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads {

    util::internal_allocator<thread_data_stackful>
        thread_data_stackful::thread_alloc_;

    thread_data_stackful::~thread_data_stackful()
    {
        LTM_(debug) << "~thread_data_stackful(" << this
                    << "), description("    //-V128
                    << this->get_description() << "), phase("
                    << this->get_thread_phase() << ")";
    }

}}    // namespace hpx::threads
