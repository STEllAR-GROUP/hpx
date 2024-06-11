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
        completion_manager_queue(parcelport* pp)
          : completion_manager_base(pp)
        {
            // LCI_queue_create(LCI_UR_DEVICE, &queue);
            // Hack for now
            LCI_queue_createx(LCI_UR_DEVICE,
                LCI_SERVER_NUM_PKTS * (size_t) config_t::ndevices, &queue);
        }

        ~completion_manager_queue()
        {
            LCI_queue_free(&queue);
        }

        LCI_comp_t alloc_completion()
        {
            return queue;
        }

        void enqueue_completion(LCI_comp_t comp)
        {
            HPX_UNUSED(comp);
        }

        LCI_request_t poll();

        LCI_comp_t get_completion_object()
        {
            return queue;
        }

    private:
        LCI_comp_t queue;
    };
}    // namespace hpx::parcelset::policies::lci

#endif
