//  Copyright (c) 2024 Jiakun Yan
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/assert.hpp>
#include <hpx/parcelport_lci/completion_manager/completion_manager_sync.hpp>
#include <hpx/parcelport_lci/parcelport_lci.hpp>

namespace hpx::parcelset::policies::lci {
    LCI_request_t completion_manager_sync::poll()
    {
        LCI_request_t request;
        request.flag = LCI_ERR_RETRY;

        LCI_comp_t sync = nullptr;
        {
            std::unique_lock l(lock, std::try_to_lock);
            if (l.owns_lock() && !sync_list.empty())
            {
                sync = sync_list.front();
                sync_list.pop_front();
            }
        }
        if (sync)
        {
            LCI_error_t ret = LCI_sync_test(sync, &request);
            if (ret == LCI_OK)
            {
                HPX_ASSERT(request.flag == LCI_OK);
                LCI_sync_free(&sync);
            }
            else
            {
                if (config_t::progress_type == config_t::progress_type_t::poll)
                    pp_->do_progress_local();
                std::unique_lock l(lock);
                sync_list.push_back(sync);
            }
        }
        return request;
    }
}    // namespace hpx::parcelset::policies::lci
