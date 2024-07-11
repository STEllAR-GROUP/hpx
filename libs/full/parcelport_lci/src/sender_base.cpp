//  Copyright (c) 2023-2024 Jiakun Yan
//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2014-2015 Thomas Heller
//  Copyright (c)      2020 Google
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_LCI)

#include <hpx/assert.hpp>

#include <hpx/modules/lci_base.hpp>
#include <hpx/parcelport_lci/backlog_queue.hpp>
#include <hpx/parcelport_lci/completion_manager_base.hpp>
#include <hpx/parcelport_lci/header.hpp>
#include <hpx/parcelport_lci/locality.hpp>
#include <hpx/parcelport_lci/parcelport_lci.hpp>
#include <hpx/parcelport_lci/receiver_base.hpp>
#include <hpx/parcelport_lci/sender_base.hpp>
#include <hpx/parcelport_lci/sender_connection_base.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace hpx::parcelset::policies::lci {
    bool sender_base::background_work(size_t /* num_thread */) noexcept
    {
        bool did_some_work = false;
        auto poll_comp_start = util::lci_environment::pcounter_now();
        auto completion_manager_p = pp_->get_tls_device().completion_manager_p;
        LCI_request_t request = completion_manager_p->send->poll();
        util::lci_environment::pcounter_add(util::lci_environment::poll_comp,
            util::lci_environment::pcounter_since(poll_comp_start));

        if (request.flag == LCI_OK)
        {
            auto useful_bg_start = util::lci_environment::pcounter_now();
            did_some_work = true;
            auto* sharedPtr_p = (connection_ptr*) request.user_context;
            HPX_ASSERT(sharedPtr_p->get());
            sender_connection_base::return_t ret = (*sharedPtr_p)->send(true);
            if (ret.status == sender_connection_base::return_status_t::done)
            {
                (*sharedPtr_p)->done();
                delete sharedPtr_p;
            }
            else if (ret.status ==
                sender_connection_base::return_status_t::wait)
            {
                completion_manager_p->send->enqueue_completion(ret.completion);
            }
            util::lci_environment::pcounter_add(
                util::lci_environment::useful_bg_work,
                util::lci_environment::pcounter_since(useful_bg_start));
        }

        return did_some_work;
    }

    bool sender_base::send_immediate(parcelset::parcelport* pp,
        parcelset::locality const& dest, parcel_buffer_type buffer,
        callback_fn_type&& callbackFn)
    {
        int dest_rank = dest.get<locality>().rank();
        auto connection = create_connection(dest_rank, pp);
        connection->buffer_ = HPX_MOVE(buffer);
        connection->async_write(HPX_MOVE(callbackFn), nullptr);
        return true;
    }
}    // namespace hpx::parcelset::policies::lci

#endif
