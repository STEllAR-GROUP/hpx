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
#include <hpx/parcelport_lci/receiver_base.hpp>
#include <hpx/parcelport_lci/sender_connection_base.hpp>

namespace hpx::parcelset::policies::lci::backlog_queue {
    thread_local backlog_queue_t tls_backlog_queue;

    void push(message_ptr message)
    {
        if (tls_backlog_queue.messages.size() <= (size_t) message->dst_rank)
        {
            tls_backlog_queue.messages.resize(message->dst_rank + 1);
        }
        auto& message_queue = tls_backlog_queue.messages[message->dst_rank];
        if (!message_queue.empty())
        {
            bool succeed = message_queue.back()->tryMerge(message);
            if (succeed)
            {
                message->done();
                return;
            }
        }
        tls_backlog_queue.messages[message->dst_rank].push_back(
            HPX_MOVE(message));
    }

    bool empty(int dst_rank)
    {
        if (tls_backlog_queue.messages.size() <= (size_t) dst_rank)
        {
            tls_backlog_queue.messages.resize(dst_rank + 1);
        }
        bool ret = tls_backlog_queue.messages[dst_rank].empty();
        return ret;
    }

    bool background_work(
        completion_manager_base* completion_manager, size_t num_thread) noexcept
    {
        bool did_some_work = false;
        for (size_t i = 0; i < tls_backlog_queue.messages.size(); ++i)
        {
            size_t idx = (num_thread + i) % tls_backlog_queue.messages.size();
            while (idx < tls_backlog_queue.messages.size() &&
                !tls_backlog_queue.messages[idx].empty())
            {
                message_ptr message = tls_backlog_queue.messages[idx].front();
                auto ret = message->send_nb();
                if (ret.status == sender_connection_base::return_status_t::done)
                {
                    tls_backlog_queue.messages[idx].pop_front();
                    message->done();
                    did_some_work = true;
                }
                else if (ret.status ==
                    sender_connection_base::return_status_t::wait)
                {
                    tls_backlog_queue.messages[idx].pop_front();
                    did_some_work = true;
                    completion_manager->enqueue_completion(ret.completion);
                }
                else
                {
                    HPX_ASSERT(ret.status ==
                        sender_connection_base::return_status_t::retry);
                    break;
                }
            }
        }
        return did_some_work;
    }
}    // namespace hpx::parcelset::policies::lci::backlog_queue

#endif
