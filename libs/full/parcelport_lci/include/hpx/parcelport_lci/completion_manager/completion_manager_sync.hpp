//  Copyright (c) 2023-2024 Jiakun Yan
//  Copyright (c) 2014-2023 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_LCI)

#include <hpx/assert.hpp>
#include <hpx/parcelport_lci/completion_manager_base.hpp>
#include <deque>

namespace hpx::parcelset::policies::lci {
    struct completion_manager_sync : public completion_manager_base
    {
        completion_manager_sync(parcelport* pp)
          : completion_manager_base(pp)
        {
        }

        ~completion_manager_sync() {}

        ::lci::comp_t alloc_completion()
        {
            ::lci::comp_t sync = ::lci::alloc_sync();
            return sync;
        }

        void free_completion(::lci::comp_t comp)
        {
            ::lci::free_comp(&comp);
        }

        void enqueue_completion(::lci::comp_t comp)
        {
            std::unique_lock l(lock);
            sync_list.push_back(comp);
        }

        ::lci::status_t poll();

        ::lci::comp_t get_completion_object()
        {
            throw std::runtime_error("completion_manager_sync does not have a "
                                     "single completion object");
        }

    private:
        hpx::spinlock lock;
        std::deque<::lci::comp_t> sync_list;
    };
}    // namespace hpx::parcelset::policies::lci

#endif
