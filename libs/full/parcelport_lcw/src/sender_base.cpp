//  Copyright (c) 2025 Jiakun Yan
//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2014-2015 Thomas Heller
//  Copyright (c)      2020 Google
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_LCW)

#include <hpx/assert.hpp>

#include <hpx/modules/lcw_base.hpp>
#include <hpx/parcelport_lcw/backlog_queue.hpp>
#include <hpx/parcelport_lcw/completion_manager_base.hpp>
#include <hpx/parcelport_lcw/header.hpp>
#include <hpx/parcelport_lcw/locality.hpp>
#include <hpx/parcelport_lcw/parcelport_lcw.hpp>
#include <hpx/parcelport_lcw/receiver_base.hpp>
#include <hpx/parcelport_lcw/sender_base.hpp>
#include <hpx/parcelport_lcw/sender_connection_base.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace hpx::parcelset::policies::lcw {
    bool sender_base::background_work(size_t /* num_thread */) noexcept
    {
        bool did_some_work = false;
        auto poll_comp_start = util::lcw_environment::pcounter_now();
        auto completion_manager_p = pp_->get_tls_device().completion_manager_p;
        ::lcw::request_t request;
        bool poll_ret = completion_manager_p->send->poll(request);
        util::lcw_environment::pcounter_add(util::lcw_environment::poll_comp,
            util::lcw_environment::pcounter_since(poll_comp_start));

        if (poll_ret)
        {
            auto useful_bg_start = util::lcw_environment::pcounter_now();
            did_some_work = true;
            auto* sharedPtr_p = (connection_ptr*) request.user_context;
            sender_connection_base::return_t ret = (*sharedPtr_p)->send();
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
            util::lcw_environment::pcounter_add(
                util::lcw_environment::useful_bg_work,
                util::lcw_environment::pcounter_since(useful_bg_start));
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
}    // namespace hpx::parcelset::policies::lcw

#endif
