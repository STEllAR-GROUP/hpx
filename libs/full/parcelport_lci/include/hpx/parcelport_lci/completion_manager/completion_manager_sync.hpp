//  Copyright (c) 2014-2023 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_LCI)

#include <hpx/parcelport_lci/completion_manager_base.hpp>

#include <hpx/assert.hpp>

namespace hpx::parcelset::policies::lci {
    struct completion_manager_sync : public completion_manager_base
    {
        completion_manager_sync() {}

        ~completion_manager_sync() {}

        LCI_comp_t alloc_completion()
        {
            LCI_comp_t sync;
            LCI_sync_create(LCI_UR_DEVICE, 1, &sync);
            return sync;
        }

        void enqueue_completion(LCI_comp_t comp)
        {
            std::unique_lock l(lock);
            sync_list.push_back(comp);
        }

        LCI_request_t poll()
        {
            LCI_request_t request;
            request.flag = LCI_ERR_RETRY;
            if (sync_list.empty())
            {
                return request;
            }
            {
                std::unique_lock l(lock, std::try_to_lock);
                if (l.owns_lock() && !sync_list.empty())
                {
                    LCI_comp_t sync = sync_list.front();
                    sync_list.pop_front();
                    LCI_error_t ret = LCI_sync_test(sync, &request);
                    if (ret == LCI_OK)
                    {
                        HPX_ASSERT(request.flag == LCI_OK);
                        LCI_sync_free(&sync);
                    }
                    else
                    {
                        sync_list.push_back(sync);
                    }
                }
            }
            return request;
        }

    private:
        hpx::spinlock lock;
        std::deque<LCI_comp_t> sync_list;
    };
}    // namespace hpx::parcelset::policies::lci

#endif
