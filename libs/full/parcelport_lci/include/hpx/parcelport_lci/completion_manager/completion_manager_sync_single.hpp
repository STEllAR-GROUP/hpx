//  Copyright (c) 2023-2024 Jiakun Yan
//  Copyright (c) 2014-2023 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_LCI)

#include <hpx/parcelport_lci/completion_manager_base.hpp>

namespace hpx::parcelset::policies::lci {
    struct completion_manager_sync_single : public completion_manager_base
    {
        completion_manager_sync_single(parcelport* pp)
          : completion_manager_base(pp)
        {
            LCI_sync_create(LCI_UR_DEVICE, 1, &sync);
        }

        ~completion_manager_sync_single()
        {
            LCI_sync_free(&sync);
        }

        LCI_comp_t alloc_completion()
        {
            return sync;
        }

        void enqueue_completion(LCI_comp_t comp)
        {
            HPX_UNUSED(comp);
            lock.unlock();
        }

        LCI_request_t poll();

    private:
        hpx::spinlock lock;
        LCI_comp_t sync;
    };
}    // namespace hpx::parcelset::policies::lci

#endif
