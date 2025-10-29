//  Copyright (c) 2024 Jiakun Yan
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/parcelport_lci/completion_manager/completion_manager_queue.hpp>
#include <hpx/parcelport_lci/parcelport_lci.hpp>

namespace hpx::parcelset::policies::lci {
    ::lci::status_t completion_manager_queue::poll()
    {
        ::lci::status_t status = ::lci::cq_pop(queue);
        if (status.is_retry())
            if (config_t::progress_type == config_t::progress_type_t::poll)
                pp_->do_progress_local();
        return status;
    }
}    // namespace hpx::parcelset::policies::lci
