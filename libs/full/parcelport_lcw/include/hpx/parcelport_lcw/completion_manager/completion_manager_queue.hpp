//  Copyright (c) 2025 Jiakun Yan
//  Copyright (c) 2014-2023 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_LCW)

#include <hpx/parcelport_lcw/completion_manager_base.hpp>

namespace hpx::parcelset::policies::lcw {
    struct completion_manager_queue : public completion_manager_base
    {
        completion_manager_queue()
        {
            queue = ::lcw::alloc_cq();
        }

        ~completion_manager_queue()
        {
            ::lcw::free_cq(queue);
        }

        ::lcw::comp_t alloc_completion()
        {
            return queue;
        }

        void enqueue_completion(::lcw::comp_t comp)
        {
            HPX_UNUSED(comp);
        }

        bool poll(::lcw::request_t& request)
        {
            return ::lcw::poll_cq(queue, &request);
        }

        ::lcw::comp_t get_completion_object()
        {
            return queue;
        }

    private:
        ::lcw::comp_t queue;
    };
}    // namespace hpx::parcelset::policies::lcw

#endif
