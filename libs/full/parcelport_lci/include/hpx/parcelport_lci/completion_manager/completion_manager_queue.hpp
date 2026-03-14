//  Copyright (c) 2023-2024 Jiakun Yan
//  Copyright (c) 2014-2023 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_LCI)

#include <hpx/parcelport_lci/config.hpp>
#include <hpx/parcelport_lci/completion_manager_base.hpp>

namespace hpx::parcelset::policies::lci {
    struct completion_manager_queue : public completion_manager_base
    {
        completion_manager_queue(parcelport* pp, bool zero_copy_am_ = false)
          : completion_manager_base(pp)
        {
            queue = ::lci::alloc_cq_x().zero_copy_am(zero_copy_am_)();
        }

        ~completion_manager_queue()
        {
            ::lci::free_comp(&queue);
        }

        ::lci::comp_t alloc_completion()
        {
            return queue;
        }

        void free_completion(::lci::comp_t comp)
        {
            HPX_UNUSED(comp);
        }

        void enqueue_completion(::lci::comp_t comp)
        {
            HPX_UNUSED(comp);
        }

        ::lci::status_t poll();

        ::lci::comp_t get_completion_object()
        {
            return queue;
        }

    private:
        ::lci::comp_t queue;
    };
}    // namespace hpx::parcelset::policies::lci

#endif
